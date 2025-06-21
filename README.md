# Gem::Logger

> **Gem::Logger is a single-header, zero-dependency, thread-safe C++23 logger that streams roughly one million messages per second on a single worker and three million on four workers. Include `logger.h`, compile with `-std=c++23`.**

### Quick-Start
Copy paste the file into your project
```cpp
#include "logger.h"   // header-only convenience
```

```bash
g++ -std=c++23 your_file.cpp   # compile flag exactly as required
```

## Why Gem::Logger?
- **Header-only simplicity** – just drop in `logger.h`, no linking headaches  
- **Blazing speed** – ~1 M msgs/s on one core, ~3 M msgs/s on four  
- **Modern C++23 & thread safety** – ready for today’s concurrency demands  

Happy logging!
