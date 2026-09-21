"""有界的精确元素组成搜索。

本模块回答的问题是：在每种元素都有上限的前提下，哪些中性分子式的理论质量
落在 ``target ± tolerance`` 窗口内？整体流程可以按下面的顺序理解：

1. ``FormulaEnumerator.search`` 把元素上限裁剪到“质量窗口内不可能超过的最大计数”；
2. ``_search`` 按固定元素顺序枚举计数，并在每个中间节点估算剩余质量；
3. ``_possible`` 查询预先构造的余数表，尽早剪掉不可能命中的分支；
4. 到最后一个元素时直接求出可行计数，而不是继续展开一层循环；
5. 返回前把搜索顺序还原成 ``ELEMENTS`` 的公共向量顺序，并用浮点精确质量复核。

整数质量表只用于保守剪枝；所有返回结果仍会用双精度单同位素质量和请求窗口
再次核验。这样既能控制高质量样本的搜索规模，又不会把整数近似当作最终答案。
"""

from functools import lru_cache
import math

import numpy as np
from numba import njit

from .chemistry import ELEMENTS, MASSES

# ``ELEMENTS`` 的公共顺序是 C/H/N/O/P/S/F/Cl/Br/I，但搜索使用这里的顺序。
# 把 H 放在搜索位置 0、最后才直接求它，是因为氢的质量最小：给定其他元素
# 后，目标质量窗口通常只对应很窄的 H 计数区间，可以省掉大量组合展开。
ORDER = np.array([1, 0, 2, 3, 6, 4, 5, 7, 8, 9])
# 整数质量的缩放因子只用于余数表剪枝，不改变最终精确质量判断。
SCALE = 100000


class SearchLimitExceeded(RuntimeError):
    """表示搜索超过资源上限，结果不能被当成完整候选列表。"""

    pass


@njit(cache=True)
def _residue_tables(weights):
    """为每个搜索前缀构造可达余数的最小整数质量表。

    ``weights`` 是按 ``ORDER`` 排列并放大后的原子质量。表的第一层记录：
    使用当前前缀元素时，每个 ``mod`` 余数至少需要多少整数质量；后续层是
    稀疏表，允许 ``_range_min`` 在一个连续余数区间内快速查询最小值。

    这里保存的是“能否到达”的下界，不保存精确的元素计数组合，因此只适合
    拒绝明显不可能的分支，不能直接作为最终候选结果。
    """
    # 以最小搜索质量（氢）作为模数。任意质量区间覆盖完整模数时，必然包含
    # 某个余数，所以后面的 ``_possible`` 可以直接放行而不查表。
    mod = weights[0]
    levels = int(math.log2(mod)) + 1
    tables = np.full((len(weights), levels, mod), np.int64(10**15), dtype=np.int64)
    # “只使用氢”时只有余数 0 的空组成可达；其他余数先标成不可达的大数。
    tables[0, 0, 0] = 0
    for i in range(1, len(weights)):
        # 继承前一个元素前缀，再把当前原子质量沿余数环传播，得到最小可达质量。
        arr = tables[i, 0]
        arr[:] = tables[i - 1, 0]
        w = weights[i]
        g = math.gcd(int(mod), int(w))
        for root in range(g):
            r = root
            for _ in range(2 * (mod // g)):
                nxt = (r + w) % mod
                if arr[r] + w < arr[nxt]:
                    arr[nxt] = arr[r] + w
                r = nxt
    # 为每个前缀构造区间最小值稀疏表。这样查询 [lo, hi] 不需要逐余数扫描。
    for i in range(len(weights)):
        for level in range(1, levels):
            step = 1 << (level - 1)
            for r in range(mod):
                tables[i, level, r] = min(
                    tables[i, level - 1, r],
                    tables[i, level - 1, min(r + step, mod - 1)],
                )
    return tables


@njit(cache=True)
def _range_min(table, lo, hi):
    """用稀疏表查询闭区间内的最小剩余质量。

    调用方保证 ``lo <= hi`` 且区间在余数表范围内；两个重叠块的最小值即可
    覆盖任意长度的闭区间。
    """
    level = int(math.log2(hi - lo + 1))
    return min(table[level, lo], table[level, hi - (1 << level) + 1])


@njit(cache=True)
def _possible(table, lo, hi, mod):
    """判断给定质量区间是否仍可能命中某个整数余数。

    ``lo``/``hi`` 是考虑整数缩放误差后得到的剩余质量区间，可能跨过模数边界，
    所以需要拆成 ``[a, mod-1]`` 和 ``[0, b]`` 两段查询。返回 ``False`` 才能
    安全剪枝；返回 ``True`` 只表示“不能证明它不可能”，后续仍会继续枚举并在
    最后用精确质量过滤。
    """
    if hi < 0:
        return False
    lo = max(0, lo)
    if hi - lo >= mod:
        return True
    a = lo % mod
    b = hi % mod
    if a <= b:
        return _range_min(table, a, b) <= hi
    return min(_range_min(table, a, mod - 1), _range_min(table, 0, b)) <= hi


@lru_cache(maxsize=1)
def _tables():
    """缓存按固定原子质量生成的剪枝表，避免每次搜索重复构建。

    原子质量和 ``ORDER`` 是模块级常量，因此同一进程内所有枚举器都可以共享
    这份只读表；元素上限和质量窗口仍在每次 ``search`` 中单独处理。
    """
    return _residue_tables(np.rint(MASSES[ORDER] * SCALE).astype(np.int64))


class FormulaEnumerator:
    """在元素上限内枚举落入质量窗口的中性分子式向量。

    ``max_results`` 是一次质量搜索的硬资源上限；超过它会抛出
    :class:`SearchLimitExceeded`，而不是返回一个看似完整的截断结果。
    """

    def __init__(self, element_limits, max_results=250000):
        """保存元素计数上限，并将缺省元素视为不可用。

        ``self.limits`` 按 ``ELEMENTS`` 的公共顺序保存；真正进入 Numba 搜索时，
        ``search`` 会再按 ``ORDER`` 重排。``max_results`` 是单次搜索的资源边界，
        不是最终向用户展示的 Top-K 或候选列表上限。
        """
        self.limits = np.array(
            [element_limits.get(e, 0) for e in ELEMENTS], dtype=np.int64
        )
        self.max_results = max_results
        if (self.limits < 0).any():
            raise ValueError("Negative element limit")

    def search(self, target, tolerance):
        """搜索目标质量窗口，并在返回前用精确质量做最终过滤。

        负质量直接返回空集；非有限输入或非正容差属于调用错误。搜索内部使用
        放大后的整数质量做剪枝，但 ``errors`` 会给剪枝区间补上舍入误差余量，
        避免近似质量把真实命中误剪掉。
        """
        if not np.isfinite(target) or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(
                "Mass and tolerance must be finite; tolerance must be positive"
            )
        if target <= 0:
            return np.empty((0, 10), dtype=np.int64)
        # 后续数组都处于搜索顺序：这样 _search 可以从 index=9 向 index=0
        # 回退，而返回给调用方前再通过 argsort(ORDER) 恢复公共元素顺序。
        masses = MASSES[ORDER]
        # 单个元素即使只使用一次，也不能超过 target+tolerance；这个上界会
        # 显著减少高质量窗口下的组合数量。
        limits = np.minimum(
            self.limits[ORDER], np.floor((target + tolerance) / masses).astype(np.int64)
        )
        weights = np.rint(masses * SCALE).astype(np.int64)
        # ``errors[i]`` 是从搜索位置 0 到 i 的最坏累计舍入误差上界。
        # _search 用它扩大整数区间，保证整数剪枝只会少剪、不误剪。
        errors = np.cumsum(limits * np.abs(weights - masses * SCALE))
        # 保持 NumPy 数组作为 API 边界，避免把 numba.typed 容器暴露给调用方。
        out = _search(
            float(target),
            float(tolerance),
            limits,
            masses,
            weights,
            _tables(),
            errors,
            self.max_results,
        )
        # _search 在发现第 max_results+1 个结果时立即返回；这里把该信号
        # 转成明确异常，调用方才能把它报告成 resource_limit，而不是误报正常结果。
        if len(out) > self.max_results:
            raise SearchLimitExceeded(
                f"Mass search exceeded {self.max_results} results; increase max_search_results explicitly"
            )
        if len(out) == 0:
            return np.empty((0, 10), dtype=np.int64)
        # Numba 内部的 counts 按 ORDER 存储；恢复到 chemistry.ELEMENTS 的协议顺序。
        result = np.asarray(out)[:, np.argsort(ORDER)]
        # 整数缩放和剪枝只保证候选“不明显不可能”，最终答案必须回到双精度质量。
        return result[np.abs(result @ MASSES - target) <= tolerance + 1e-9]


@njit(cache=True)
def _search(target, tol, limits, masses, weights, tables, errors, max_results):
    """使用显式栈遍历元素计数，并在每层用余数表提前剪枝。

    这是一个手写的深度优先搜索，等价于递归枚举但不依赖递归调用。各数组含义：

    - ``counts``：当前路径在搜索顺序下的元素计数；
    - ``next_count``：每层下一次要尝试的计数，承担递归循环变量的作用；
    - ``remaining``：进入某层时还需要解释的精确质量；
    - ``index``：当前正在尝试的搜索位置，向 0 下降表示深入，向 9 上升表示回溯。

    当 ``index > 0`` 时，先尝试当前元素计数 ``k``，再用余数表判断剩余元素是否
    还有可能命中窗口。只有 ``_possible`` 返回真才向下一层深入；否则继续尝试
    当前层的下一个 ``k``。到 ``index == 0`` 时，氢是最后变量，可以直接从剩余
    质量反解计数区间，并把有效非零向量加入结果。
    """
    # Numba 对普通 Python list 的类型推断要求先放入同类型数组再弹出；外部
    # API 不会看到这个临时容器，最终由 FormulaEnumerator 转成 NumPy 数组。
    out = [np.zeros(10, dtype=np.int64)]
    out.pop()
    # 当前计数、每层循环游标和每层剩余质量共同保存 DFS 的完整状态。
    counts = np.zeros(10, dtype=np.int64)
    next_count = np.zeros(10, dtype=np.int64)
    remaining = np.zeros(10)
    remaining[9] = target
    index = 9
    while index < 10:
        if index == 0:
            # H 是最后求解的变量。由于其它元素计数已经固定，只需计算窗口
            # [remaining[0]-tol, remaining[0]+tol] 对应的整数 H 范围。
            low = max(0, int(math.ceil((remaining[0] - tol - 1e-10) / masses[0])))
            high = min(
                limits[0], int(math.floor((remaining[0] + tol + 1e-10) / masses[0]))
            )
            for h in range(low, high + 1):
                counts[0] = h
                # 排除全零向量：项目要求返回真实的中性分子式，而不是空组成。
                if np.sum(counts) > 0:
                    out.append(counts.copy())
                # 尽早停止并把“结果不完整”交给上层处理，不能静默截断。
                if len(out) > max_results:
                    return out
            counts[0] = 0
            # 当前层穷举完成，向上回溯；上层会递增自己的 next_count。
            index += 1
            continue
        k = next_count[index]
        # 计数超过元素上限，或仅当前元素就已经超过剩余质量时，当前层无须
        # 再尝试更大的 k，直接回溯到上一层。
        if k > limits[index] or k * masses[index] > remaining[index] + tol:
            counts[index] = 0
            index += 1
            continue
        # 先递增游标，确保从子层回溯后不会重复尝试同一个 k。
        next_count[index] += 1
        rem = remaining[index] - k * masses[index]
        # 余数表使用放大后的整数质量；errors[index-1] 是更低位置元素可能
        # 产生的累计舍入误差，因此这里扩大区间而不是缩小区间。
        lo_i = int(math.floor((rem - tol) * SCALE - errors[index - 1] - 1))
        hi_i = int(math.ceil((rem + tol) * SCALE + errors[index - 1] + 1))
        if _possible(tables[index - 1], lo_i, hi_i, weights[0]):
            # 当前 k 仍可能命中：保存路径、保存下一层目标质量，并清空下一层
            # 的循环游标，然后下降一层继续 DFS。
            counts[index] = k
            remaining[index - 1] = rem
            next_count[index - 1] = 0
            index -= 1
        # 如果不可能，index 不变；下一轮会用递增后的 next_count[index]
        # 尝试当前元素的下一个计数。
    return out
