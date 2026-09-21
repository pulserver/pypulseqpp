# Prerequisites and supported platforms

pypulseqpp supports Python 3.10 through 3.13. CI tests the lower and upper
bounds on Linux, macOS and Windows.

Published wheels cover:

| Platform | Architectures |
| --- | --- |
| Linux | x86-64, glibc (`manylinux`) |
| macOS | Apple silicon and x86-64 |
| Windows | AMD64 |

The native core is compiled into each wheel. Source builds may work on other
C++17 platforms, but those configurations are not release-tested. A source
build requires a C++17 compiler, CMake, Python development headers and the
checked-out Git submodules.
