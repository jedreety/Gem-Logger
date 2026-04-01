# Gem::Logger — Documentation

> Thread-safe, header-only C++23 logging system.
> Lock-free queue, 7 levels, color DSL, JSON output, file rotation, `@{key}` interpolation.

---

## Quick Start

```cpp
#define GEMLOG_SIMPLE_HANDLER_CONSOLE   // auto-register console handler
#include "logger.h"

int main() {
    LOG_INFO("Server started on port @{port}", {{"port", 8080}});
    LOG_WARNING("Disk usage at @{pct}%", {{"pct", 92.5}});
}
```

Output: `[INFO] 14:32:01 | Server started on port 8080 (main.cpp:4)`

---

## Compile-Time Configuration

### `GEMLOG_LEVEL` — Strip macros at compile time

| Value | Active macros | Stripped |
|-------|--------------|----------|
| `0` | All | — |
| `1` | Debug, Info, Success, Warning, Error, Critical | Trace |
| `2` | Warning, Error, Critical | Trace, Debug, Info, Success |
| `3` | Critical only | All except Critical |
| `4` | None | All |

```cpp
#define GEMLOG_LEVEL 3
#include "logger.h"
```

### `GEMLOG_SIMPLE_HANDLER_CONSOLE` — Zero-config console output

Automatically registers a colored console handler at startup (level Trace, stdout).
No manual `add_handler()` call needed.

```cpp
#define GEMLOG_SIMPLE_HANDLER_CONSOLE
#include "logger.h"
```

---

## Macros

```cpp
LOG_TRACE(message [, context] [, handler_hint])
LOG_DEBUG(message [, context] [, handler_hint])
LOG_INFO(message [, context] [, handler_hint])
LOG_SUCCESS(message [, context] [, handler_hint])
LOG_WARNING(message [, context] [, handler_hint])
LOG_ERROR(message [, context] [, handler_hint])
LOG_CRITICAL(message [, context] [, handler_hint])
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | `std::string_view` | Log message (supports `@{key}` interpolation) |
| `context` | `Gem::ContextMap` | Key-value pairs for interpolation and structured output |
| `handler_hint` | `std::string_view` | Route to a specific handler by name |

```cpp
// Simple
LOG_INFO("Application started");

// With context (interpolated)
LOG_ERROR("Connection to @{host} failed after @{retries} attempts", {
    {"host", std::string("db.local")},
    {"retries", 3}
});

// Routed to a specific handler
LOG_DEBUG("SQL query executed", {{"query_ms", 42}}, "database");
```

---

## Message Interpolation

Use `@{key}` inside messages. Values come from the `ContextMap`.

```cpp
LOG_INFO("User @{name} logged in from @{ip}", {
    {"name", std::string("alice")},
    {"ip", std::string("192.168.1.1")}
});
// → "User alice logged in from 192.168.1.1"
```

Use `@@` for a literal `@`. Missing keys are kept verbatim: `@{unknown}` → `@{unknown}`.

**ContextValue types:** `bool`, `int`, `long long`, `double`, `std::string`.

---

## Handler Configuration (Builder)

Handlers define where and how logs are written. Register via the type-safe builder:

```cpp
Gem::Logger::instance().add_handler(
    Gem::ConfigTemplate::builder()
        .name("console")                        // 1. Name (required, unique)
        .level(Gem::LogLevel::Info)              // 2. Minimum level
        .format("fmt1",                          // 3. Named format pattern
            "<bold>[%(levelname)]</bold> %(time) | %(message)")
        .output("fmt1", Gem::StreamTarget::cout()) // 4. Bind format → output
        .build()
);
```

Builder chain order: `name` → `level` → `format` → `output` → `build()`.
Multiple `.format()` + `.output()` pairs allowed per handler.

### Optional builder methods

| Method | Description |
|--------|-------------|
| `.filter(fn)` | `std::function<bool(const LogRecord&)>` — reject records |
| `.context(map)` | Base context merged into every record |
| `.structured(true)` | Emit JSON instead of formatted text |

```cpp
// Filter: only records with context key "module" == "auth"
.filter([](const Gem::LogRecord& r) {
    auto it = r.context.find("module");
    return it != r.context.end() && std::get<std::string>(it->second) == "auth";
})
```

---

## Format Tokens

Used inside format patterns as `%(token)`.

| Token | Output | Example |
|-------|--------|---------|
| `%(levelname)` | Level name | `WARNING` |
| `%(date)` | Date | `2025-06-15` |
| `%(time)` | Time | `14:32:01` |
| `%(message)` | Interpolated message | `Server started` |
| `%(thread)` | Thread ID | `140234` |
| `%(file)` | Source file | `main.cpp` |
| `%(line)` | Line number | `42` |
| `%(function)` | Function name | `main` |
| `%(context[key])` | Specific context value | `%(context[user_id])` |

```cpp
.format("detailed",
    "[%(levelname)] %(date) %(time) [%(thread)] %(file):%(line) - %(message)")
```

---

## Color DSL

Wrap format text in HTML-like tags. Applied only to stream outputs (not files/JSON).

| Tag | Effect |
|-----|--------|
| `<red>` `</red>` | Red text |
| `<green>` `</green>` | Green text |
| `<yellow>` `</yellow>` | Yellow text |
| `<blue>` `</blue>` | Blue text |
| `<magenta>` `</magenta>` | Magenta text |
| `<cyan>` `</cyan>` | Cyan text |
| `<white>` `</white>` | White text |
| `<bold>` `</bold>` | Bold |
| `<dim>` `</dim>` | Dim/faint |
| `<italic>` `</italic>` | Italic |
| `<underline>` `</underline>` | Underline |

Tags are nestable:

```cpp
.format("styled",
    "<bold><red>[%(levelname)]</red></bold> %(time) | %(message) <dim>(%(file):%(line))</dim>")
```

---

## Output Targets

### Stream (console)

```cpp
.output("fmt", Gem::StreamTarget::cout())   // stdout
.output("fmt", Gem::StreamTarget::cerr())   // stderr
```

### File (with auto-rotation)

```cpp
.output("fmt", std::filesystem::path("logs/app.log"))
```

File features (via `FileCache`):
- Auto-creates directories
- Rotates at 100 MB (configurable)
- Idle file handles closed after 30s
- Max 64 cached files

---

## JSON / Structured Output

```cpp
Gem::Logger::instance().add_handler(
    Gem::ConfigTemplate::builder()
        .name("json_file")
        .level(Gem::LogLevel::Info)
        .format("json", "unused")           // pattern ignored when structured
        .output("json", std::filesystem::path("logs/app.jsonl"))
        .structured(true)
        .build()
);
```

Output per line:
```json
{"timestamp":1718450000,"level":"INFO","message":"User logged in","thread":"140234","file":"main.cpp","line":42,"function":"handle_login","context":{"user_id":123}}
```

---

## Multi-Handler Example

```cpp
#define GEMLOG_LEVEL 0
#include "logger.h"

int main() {
    auto& logger = Gem::Logger::instance();

    // Console: colored, warnings+
    logger.add_handler(
        Gem::ConfigTemplate::builder()
            .name("console")
            .level(Gem::LogLevel::Warning)
            .format("c", "<bold><yellow>[%(levelname)]</yellow></bold> %(message)")
            .output("c", Gem::StreamTarget::cerr())
            .build()
    );

    // File: all levels, detailed
    logger.add_handler(
        Gem::ConfigTemplate::builder()
            .name("file")
            .level(Gem::LogLevel::Trace)
            .format("f", "[%(levelname)] %(date) %(time) [%(thread)] %(message) (%(file):%(line))")
            .output("f", std::filesystem::path("logs/app.log"))
            .build()
    );

    // JSON: structured, errors only
    logger.add_handler(
        Gem::ConfigTemplate::builder()
            .name("json")
            .level(Gem::LogLevel::Error)
            .format("j", "")
            .output("j", std::filesystem::path("logs/errors.jsonl"))
            .structured(true)
            .build()
    );

    LOG_TRACE("Only in file");
    LOG_WARNING("Console + file");
    LOG_ERROR("All three handlers", {{"code", 500}});
}
```

---

## Advanced

### Overflow Policy

```cpp
Gem::Logger::instance().set_overflow_policy(Gem::OverflowPolicy::DropNewest);
```

| Policy | Behavior when queue is full |
|--------|---------------------------|
| `Block` | Caller waits *(default)* |
| `DropOldest` | Discard oldest queued record |
| `DropNewest` | Discard the new record |

### Stats

```cpp
auto s = Gem::Logger::instance().get_stats();
// s.queued, s.dropped, s.processed, s.handlers
```

### Multi-Worker Logger

```cpp
using FastLogger = Gem::MultiWorkerLogger<4>;  // 4 consumer threads
```

Uses lock-free queue (MPMC) instead of `std::deque` + mutex.

### Remove Handler

```cpp
Gem::Logger::instance().remove_handler("console");
```

### Shutdown

```cpp
Gem::Logger::instance().shutdown();  // drains queue, joins workers
```

Called automatically on destruction.

---

## Log Levels

| Level | Value | Typical use |
|-------|-------|-------------|
| `Trace` | 0 | Granular debugging (loop iterations, variable dumps) |
| `Debug` | 1 | Development diagnostics |
| `Info` | 2 | Normal operational events |
| `Success` | 3 | Completed operations |
| `Warning` | 4 | Recoverable issues |
| `Error` | 5 | Failures requiring attention |
| `Critical` | 6 | System-level failures |
