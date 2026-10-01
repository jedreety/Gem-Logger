# Gem Logger guide

[Back to the README](../README.md)

Everything `logger.h` can do, with every option.

```cpp
#include "logger.h"

int main() {
    int port = 8080;
    LOG_INFO("Server listening on port {}", port);
    LOG_WARNING("Disk usage at {:.1f}%", 92.5);
}
```

```
14:32:01.123 INFO     Server listening on port 8080 (main.cpp:5)
14:32:01.123 WARNING  Disk usage at 92.5% (main.cpp:6)
```

Nothing to set up: logs go to a colored console by default. Copy `logger.h` into your project and include it.

**Requirements:** C++23. MSVC 19.37+ (Visual Studio 2022 17.7) with `/std:c++latest` and `/utf-8`, GCC 13+ or Clang 17+ with `-std=c++23`.

---

## Messages and variables

Messages use the [`std::format`](https://en.cppreference.com/w/cpp/utility/format/spec) syntax. A wrong format string, or a missing argument, is a compile error.

```cpp
LOG_INFO("Player {} joined ({} players)", name, count);
LOG_DEBUG("Position {:.2f}, {:.2f}", x, y);
LOG_INFO("Literal braces: {{}}");
```

Any value can be logged:

| Value | Printed as |
|-------|------------|
| types with a `std::formatter` (numbers, strings, containers, chrono...) | their standard format: `[1, 2, 3]`, `{"a": 1}` |
| types with an `operator<<` | what `operator<<` writes |
| `std::exception` and derived | `what()` |
| `std::filesystem::path` | its UTF-8 text |
| enums | their value |
| pointers | their address, `nullptr` C strings as `(null)` |

A message that is not a string literal is written as it is, braces included:

```cpp
std::string line = read_line();          // may contain { }
LOG_INFO(line);
LOG_INFO(Gem::Log::runtime_format(fmt), a, b);   // a format string known only at run time
```

### `LOG_DUMP`: names and values

```cpp
LOG_DUMP(x, name, items, items.size());
// DEBUG    x = 42, name = "bob", items = [1, 2, 3], items.size() = 3
```

In JSON output, these variables become structured fields (`"vars":{"x":42,...}`).

## Print

`Gem::print` and `Gem::println` work like `std::print`, with [color tags](#color-tags) and no log decoration.
They never mix with a log line, and they stay in order with the logs, even in async mode.

```cpp
Gem::println("<green>Build succeeded</green> in {:.2f} s", seconds);
Gem::print("Progress: {}%\r", percent);
Gem::println();
```

## Levels

| Level | Macro | Use |
|-------|-------|-----|
| `Trace` | `LOG_TRACE` | Very detailed steps |
| `Debug` | `LOG_DEBUG` | Development diagnostics |
| `Info` | `LOG_INFO` | Normal events |
| `Success` | `LOG_SUCCESS` | Completed operations |
| `Warning` | `LOG_WARNING` | Recoverable problems |
| `Error` | `LOG_ERROR` | Failures |
| `Critical` | `LOG_CRITICAL` | The program cannot go on |

```cpp
Gem::Log::set_level(Gem::Log::Level::Warning);   // global minimum, at run time
```

The arguments of a macro are not evaluated when its level is off. To guard an expensive computation:

```cpp
if (Gem::Log::enabled(Gem::Log::Level::Debug)) LOG_DEBUG("{}", expensive_report());
```

`Gem::Log::parse_level("warn")` reads a level from a command line or a config file.

### Compile-time removal

```cpp
#define GEMLOG_LEVEL GEMLOG_LEVEL_WARNING   // or a number: 0 TRACE ... 6 CRITICAL, 7 removes everything
#include "logger.h"
```

The macros below that level are compiled out: no code, no evaluated argument.

### Other macros

```cpp
LOG_WARNING_ONCE("Texture {} missing", name);   // every level has an _ONCE version: first call only
LOG_TIMER("load level");                        // DEBUG at the end of the scope: "load level: 12.48 ms"
LOG_CONTEXT("request", id);                     // see Context
```

Every macro has a function equivalent, which also records the file and line: `Gem::Log::info("...", args)`,
`Gem::Log::log(level, "...", args)`, `Gem::Log::Timer`, `Gem::Log::Context`.
Define `GEMLOG_NO_MACROS` before the include to get the functions only (for example when `<syslog.h>` already
defines `LOG_INFO`).

## Outputs

### Console

Created at start-up on `stdout`. Colors are on when the output is a terminal (the Windows console is set up for them),
and off when it is redirected or when `NO_COLOR` is set. `FORCE_COLOR=1` forces them.

```cpp
auto console = Gem::Log::console();
console->set_level(Gem::Log::Level::Info);
console->set_stream(stderr);
console->set_color(Gem::Log::ColorMode::Never);   // Auto, Always, Never
Gem::Log::remove_sink(console);                    // no console at all
```

### Files

```cpp
Gem::Log::add_file("logs/app.log");
Gem::Log::add_file("logs/debug.log", {.max_size = 50 * 1024 * 1024, .max_files = 3, .truncate = true});
```

| Option | Default | |
|--------|---------|---|
| `max_size` | 10 MB | Rotate before the file grows past this size (`0` = never) |
| `max_files` | 5 | Rotated files kept: `app.1.log` (newest) ... `app.5.log` |
| `truncate` | `false` | Start with an empty file instead of appending |
| `json` | `false` | JSON Lines instead of the text pattern |

Directories are created when needed. On Windows the file can be read by other programs while it is written.
If it cannot be opened or rotated (for example when another program holds it), the error is reported once on
`stderr` and logging goes on.

### JSON Lines

```cpp
Gem::Log::add_file("logs/app.jsonl", {.json = true});
```

```json
{"time":"2026-10-01T12:32:01.123Z","level":"ERROR","message":"db down","thread":"21436","file":"C:\\src\\db.cpp","line":42,"function":"Db::connect","channel":"db","context":{"request":"r-1"},"vars":{"retries":3}}
```

Time is in UTC. The output is always valid JSON: control characters are escaped and invalid UTF-8 is replaced.

### Callbacks and custom sinks

```cpp
// In-game console, IDE output window, tests...
Gem::Log::add_callback([](const Gem::Log::Record& record, std::string_view line) {
    game_console.add(record.level, std::string(line));
});

// Visual Studio "Output" window (declare OutputDebugStringA or include <windows.h>)
Gem::Log::add_callback([](const Gem::Log::Record&, std::string_view line) {
    OutputDebugStringA((std::string(line) + '\n').c_str());
});
```

A custom sink derives from `Gem::Log::Sink`. `write()` is always called under the logger's lock: a sink needs no
lock of its own.

```cpp
struct Overlay : Gem::Log::Sink {
    void write(const Gem::Log::Record& record, std::string_view line) override { /* ... */ }
    void flush() override { /* optional */ }
};
Gem::Log::add_sink(std::make_shared<Overlay>());
```

Every sink has a level, a pattern, a filter and a JSON switch:

```cpp
auto errors = Gem::Log::add_file("logs/errors.log");
errors->set_level(Gem::Log::Level::Error);
errors->set_pattern("%(date) %(time) %(message)");
errors->set_filter([](const Gem::Log::Record& r) { return r.channel != "net"; });
errors->set_json(true);
```

## Patterns

```cpp
Gem::Log::set_pattern("<dim>%(time)</dim> <level>%(level:<8)</level> %(message)");   // every text sink
Gem::Log::console()->set_pattern("[%(level)] %(message)");                             // one sink
```

| Token | Output |
|-------|--------|
| `%(level)` | `WARNING` |
| `%(message)` | the message |
| `%(date)` | `2026-10-01` |
| `%(time)` | `14:32:01.123` |
| `%(elapsed)` | seconds since start-up: `12.345` |
| `%(thread)` | name given by `Gem::Log::set_thread_name("render")`, or the thread id |
| `%(file)` / `%(path)` | `main.cpp` / full path |
| `%(line)` | `42` |
| `%(function)` | `Game::update` (return type and parameters removed) |
| `%(channel)` | channel name, empty for the default one |
| `%(context)` / `%(context[key])` | `request=42 user=bob` / one value |

- A token takes a `std::format` spec: `%(level:<8)` (padded), `%(line:>4)`, `%(elapsed:.6f)`.
- `%[ ... %]` is written only when the tokens inside are not empty: `%[[%(channel)] %]` gives `[net] ` or nothing.
- `%%` is a literal `%`.
- A wrong pattern throws `std::invalid_argument` at once, never at log time.

Default patterns (`Gem::Log::default_console_pattern`, `Gem::Log::default_file_pattern`):

```
console  <dim>%(time)</dim> <level>%(level:<8)</level> %[<dim>[%(channel)]</dim> %]%(message)%[ <dim>{%(context)}</dim>%] <dim>(%(file):%(line))</dim>
file     %(date) %(time) %(level:<8) [%(thread)] %[[%(channel)] %]%(message)%[ {%(context)}%] (%(file):%(line))
```

## Color tags

Tags work in patterns, in log messages and in prints. They are turned into ANSI codes on a colored console and
removed everywhere else (files, JSON, callbacks, redirected output).

| Tags | |
|------|---|
| `<red>` `<green>` `<yellow>` `<blue>` `<magenta>` `<cyan>` `<white>` `<black>` `<gray>` | text color |
| `<on_red>` `<on_green>` ... | background |
| `<bold>` `<dim>` `<italic>` `<underline>` | style |
| `<level>` | color of the record's level (patterns only) |

Close with `</red>` or `</>`. Tags nest correctly, and unknown tags stay as text (`vector<int>` is safe).
Only the format string is read for tags: a value such as a user name containing `<red>` is printed as it is.

```cpp
LOG_SUCCESS("Loaded <green>{}</green> textures", count);
Gem::Log::set_level_style(Gem::Log::Level::Info, "bold blue");
```

## Channels

A channel is a named logger for a subsystem, with its own level. Channels with the same name share their settings.

```cpp
Gem::Log::Channel net{"net"};
net.set_level(Gem::Log::Level::Warning);
net.info("Connected to {}", host);    // filtered out
net.error("Connection lost");         // 14:32:01.123 ERROR    [net] Connection lost (client.cpp:88)
```

## Context

Adds a key and a value to every record of the current thread, as long as the object lives.

```cpp
void handle(const Request& request) {
    LOG_CONTEXT("request", request.id);
    LOG_INFO("Started");   // 14:32:01.123 INFO     Started {request=42} (server.cpp:12)
}
```

In JSON output, context values become structured fields (`"context":{"request":42}`).

## Async mode

By default the calling thread writes each record and flushes it before returning: nothing is lost if the program
crashes, and the output stays in order with your own prints.

In async mode a background thread does the writing, and logging costs a fraction of a microsecond.

```cpp
Gem::Log::set_async(true);
Gem::Log::set_async(true, {.capacity = 65536, .overflow = Gem::Log::Overflow::DropNewest});
Gem::Log::flush();            // waits until everything logged so far is written
Gem::Log::set_async(false);   // writes what is queued and stops the thread
```

| Option | Default | |
|--------|---------|---|
| `capacity` | 8192 | Records waiting to be written |
| `overflow` | `Block` | When the queue is full: `Block` waits, `DropNewest` or `DropOldest` lose a record |
| `flush_level` | `Error` | Records at or above wait until written, so the last errors before a crash are on disk |

What is queued is written when the program exits normally. `Gem::Log::stats()` returns the number of records
logged, dropped and queued.

## Thread safety and guarantees

- Every function can be called from any thread. Lines never mix.
- Logging never throws. A formatting error at run time (a dynamic width out of range, a throwing `operator<<`)
  gives `message [format error: ...]`.
- Logging from inside a sink, a filter or a callback is dropped instead of dead-locking.
- Logging still works in static destructors, after `main` has returned.
- UTF-8 text is shown correctly by the Windows console (compile with `/utf-8`).

## Migrating from the previous version

| Before | Now |
|--------|-----|
| `LOG_INFO("Port @{port}", {{"port", 8080}})` | `LOG_INFO("Port {}", 8080)` |
| `#define GEMLOG_SIMPLE_HANDLER_CONSOLE` | nothing: the console is there by default |
| `Logger::instance().add_handler(ConfigTemplate::builder()...build())` | `Gem::Log::add_file(...)`, `Gem::Log::console()`, `Gem::Log::add_callback(...)` |
| `.format(...)` + `.output(...)` | `sink->set_pattern(...)`; `%(levelname)` still works, `%(time)` now has milliseconds |
| `.structured(true)` | `{.json = true}` or `sink->set_json()` |
| `.filter(fn)` | `sink->set_filter(fn)` |
| `.context(map)` | `Gem::Log::Context` or `LOG_CONTEXT` |
| handler hint (3rd argument) | `Gem::Log::Channel` and `sink->set_filter(...)` |
| `set_overflow_policy(...)` | `Gem::Log::set_async(true, {.overflow = ...})` |
| `get_stats()` / `shutdown()` | `Gem::Log::stats()` / `Gem::Log::flush()`, automatic at exit |
| `MultiWorkerLogger<N>` | `Gem::Log::set_async(true)`: one writer thread keeps the order of the records |
| `GEMLOG_LEVEL` 0 to 4 | `GEMLOG_LEVEL` = the level number, 0 `TRACE` to 6 `CRITICAL`, 7 removes everything |
| `Gem::LogLevel` | still available, same as `Gem::Log::Level` |

## License

MIT, see [LICENSE](../LICENSE).
