"""允许通过 ``python -m casmi`` 进入命令行入口。"""

from .cli import main

raise SystemExit(main())
