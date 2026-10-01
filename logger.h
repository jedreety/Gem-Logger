#pragma once

// =====================================================================================
//  Gem::Log - single-header logging library for C++23                     MIT License
// =====================================================================================
//
//  #include "logger.h"      C++23: MSVC 19.37+ (compile with /utf-8), GCC 13+, Clang 17+
//
//      LOG_INFO("Server listening on port {}", port);        // std::format syntax, checked at compile time
//      LOG_ERROR("Cannot load {}: {}", path, error);         // any type with a std::formatter or an operator<<
//      LOG_DUMP(x, name, items);                             // x = 42, name = "bob", items = [1, 2, 3]
//      Gem::println("<green>Done</green> in {:.2f} s", t);   // print with color tags, no log decoration
//
//  Nothing to set up: logs go to a colored console by default.
//      14:32:01.123 INFO     Server listening on port 8080 (main.cpp:12)
//
//  LEVELS       TRACE  DEBUG  INFO  SUCCESS  WARNING  ERROR  CRITICAL
//
//  MACROS       LOG_<LEVEL>(fmt, args...)         arguments are not evaluated when the level is off
//               LOG_<LEVEL>_ONCE(fmt, args...)    logs only the first time this line runs
//               LOG_DUMP(a, b, ...)               names and values of variables (DEBUG)
//               LOG_TIMER("label")                time spent until the end of the scope (DEBUG)
//               LOG_CONTEXT("key", value)         adds key=value to the logs of this thread in this scope
//
//  FUNCTIONS    Gem::Log::info(fmt, args...)      same as the macros, file and line included
//               Gem::print / Gem::println         std::print with color tags, kept in order with the logs
//               Gem::Log::Channel net{"net"};     named logger with its own level: net.warning("...")
//
//  CONFIGURATION (thread-safe, all optional)
//      Gem::Log::set_level(Gem::Log::Level::Debug);                    global minimum level
//      Gem::Log::set_pattern("%(time) %(level:<8) %(message)");        pattern of every text sink
//      Gem::Log::add_file("logs/app.log");                             file, rotated at 10 MB, 5 files kept
//      Gem::Log::add_file("logs/app.jsonl", {.json = true});           JSON Lines
//      Gem::Log::add_callback([](const Gem::Log::Record& r, std::string_view line) { ... });
//      Gem::Log::console()->set_level(Gem::Log::Level::Warning);       every sink: level, pattern, filter
//      Gem::Log::set_async(true);                                      writes on a background thread
//      Gem::Log::flush();                                              waits until everything is written
//
//  PATTERN TOKENS
//      %(level) %(message) %(date) %(time) %(elapsed) %(thread) %(file) %(path) %(line) %(function)
//      %(channel) %(context) %(context[key])     with an optional std::format spec: %(level:<8)
//      %[ ... %]   written only when the tokens inside are not empty: %[[%(channel)] %]
//      %%          a literal %
//
//  COLOR TAGS  in patterns and in the format strings of logs and prints
//      <red> <green> <yellow> <blue> <magenta> <cyan> <white> <black> <gray>  <on_red> (background)...
//      <bold> <dim> <italic> <underline>   <level> (patterns: color of the level)   closed by </red> or </>
//      Removed when the output is not a terminal, or when NO_COLOR is set.
//
//  COMPILE-TIME OPTIONS (define before the include)
//      GEMLOG_LEVEL n      removes the macros below level n (0 TRACE ... 6 CRITICAL, 7 removes them all)
//      GEMLOG_NO_MACROS    defines no macro (to avoid clashes, with <syslog.h> for example)
//
// =====================================================================================

#include <version>

#if !defined(__cpp_lib_format) || !defined(__cpp_lib_to_underlying)
#   error "Gem::Log needs C++23: compile with /std:c++latest (MSVC) or -std=c++23 (GCC, Clang)"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <exception>
#include <filesystem>
#include <format>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <source_location>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__cpp_lib_print)
#   include <print>
#endif

#if defined(_WIN32)
#   include <io.h>
#   include <share.h>
// Console API, declared here to keep <windows.h> out of the user's code (same signatures as <windows.h>).
extern "C" {
__declspec(dllimport) int __stdcall GetConsoleMode(void* console, unsigned long* mode);
__declspec(dllimport) int __stdcall SetConsoleMode(void* console, unsigned long mode);
}
#else
#   include <unistd.h>
#   if defined(__linux__)
#       include <sys/syscall.h>
#   endif
#endif

namespace Gem::Log {

// -- Public types ---------------------------------------------------------------------

enum class Level : std::uint8_t { Trace, Debug, Info, Success, Warning, Error, Critical, Off };

[[nodiscard]] constexpr std::string_view to_string(Level level) noexcept {
    constexpr std::string_view names[] = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL", "OFF"};
    const auto index = static_cast<std::size_t>(level);
    return index < std::size(names) ? names[index] : std::string_view("UNKNOWN");
}

// "debug", "WARNING", "warn", "err", "fatal", "off", "3"... (case-insensitive), for command lines and config files.
[[nodiscard]] constexpr std::optional<Level> parse_level(std::string_view text) noexcept {
    constexpr std::pair<std::string_view, Level> names[] = {
        {"trace", Level::Trace},     {"debug", Level::Debug}, {"info", Level::Info},         {"success", Level::Success},
        {"warning", Level::Warning}, {"warn", Level::Warning}, {"error", Level::Error},      {"err", Level::Error},
        {"critical", Level::Critical}, {"fatal", Level::Critical}, {"off", Level::Off},     {"none", Level::Off}};
    for (const auto& [name, level] : names) {
        bool same = name.size() == text.size();
        for (std::size_t i = 0; same && i < text.size(); ++i) {
            const char c = text[i] >= 'A' && text[i] <= 'Z' ? static_cast<char>(text[i] - 'A' + 'a') : text[i];
            same = c == name[i];
        }
        if (same) return level;
    }
    if (text.size() == 1 && text[0] >= '0' && text[0] <= '7') return static_cast<Level>(text[0] - '0');
    return std::nullopt;
}

enum class FieldKind : std::uint8_t { String, Number, Bool, Other };

// A key/value attached to a record: by Gem::Log::Context (context) or by LOG_DUMP (vars).
struct Field {
    std::string_view key;
    std::string_view value;
    FieldKind kind = FieldKind::String;
};

// What a sink receives. Views are valid only during the call.
struct Record {
    Level level = Level::Info;
    std::chrono::system_clock::time_point time{};
    std::string_view message;        // formatted message, color tags removed
    std::string_view ansi_message;   // the same with its color tags turned into ANSI codes
    std::string_view channel;        // empty for the default channel
    std::string_view thread;         // name given to set_thread_name(), or the thread id
    std::source_location location{};
    std::span<const Field> context;  // Gem::Log::Context values of the logging thread
    std::span<const Field> vars;     // variables of LOG_DUMP
};

enum class ColorMode : std::uint8_t { Auto, Always, Never };

// What set_async() does when its queue is full.
enum class Overflow : std::uint8_t { Block, DropNewest, DropOldest };

struct AsyncOptions {
    std::size_t capacity = 8192;          // records waiting to be written
    Overflow overflow = Overflow::Block;
    Level flush_level = Level::Error;     // records at or above wait until written (nothing lost on a crash)
};

struct FileOptions {
    std::size_t max_size = 10 * 1024 * 1024;   // rotate before the file grows past this size (0 = never)
    std::size_t max_files = 5;                 // rotated files kept: app.1.log (newest) ... app.5.log
    bool truncate = false;                     // start with an empty file instead of appending
    bool json = false;                         // JSON Lines instead of the text pattern
};

struct Stats {
    std::uint64_t logged = 0;    // records written (or queued) since the start
    std::uint64_t dropped = 0;   // records lost: full queue, or logged from inside a sink
    std::size_t queued = 0;      // records waiting in async mode
};

inline constexpr std::string_view default_console_pattern =
    "<dim>%(time)</dim> <level>%(level:<8)</level> %[<dim>[%(channel)]</dim> %]%(message)"
    "%[ <dim>{%(context)}</dim>%] <dim>(%(file):%(line))</dim>";

inline constexpr std::string_view default_file_pattern =
    "%(date) %(time) %(level:<8) [%(thread)] %[[%(channel)] %]%(message)%[ {%(context)}%] (%(file):%(line))";

namespace detail {

// -- Color tags -----------------------------------------------------------------------

struct Style {
    std::string_view name;
    std::string_view code;
};

inline constexpr Style styles[] = {
    {"bold", "\x1b[1m"},       {"dim", "\x1b[2m"},         {"italic", "\x1b[3m"},       {"underline", "\x1b[4m"},
    {"black", "\x1b[30m"},     {"red", "\x1b[31m"},        {"green", "\x1b[32m"},       {"yellow", "\x1b[33m"},
    {"blue", "\x1b[34m"},      {"magenta", "\x1b[35m"},    {"cyan", "\x1b[36m"},        {"white", "\x1b[37m"},
    {"gray", "\x1b[90m"},      {"grey", "\x1b[90m"},
    {"on_black", "\x1b[40m"},  {"on_red", "\x1b[41m"},     {"on_green", "\x1b[42m"},    {"on_yellow", "\x1b[43m"},
    {"on_blue", "\x1b[44m"},   {"on_magenta", "\x1b[45m"}, {"on_cyan", "\x1b[46m"},     {"on_white", "\x1b[47m"},
};

inline constexpr std::string_view ansi_reset = "\x1b[0m";

[[nodiscard]] constexpr const Style* find_style(std::string_view name) noexcept {
    for (const auto& style : styles)
        if (style.name == name) return &style;
    return nullptr;
}

// <name>, </name> or </>. Unknown names are plain text, so "vector<int>" or "a < b" stay as they are.
struct Tag {
    std::size_t length = 0;   // '<' and '>' included
    bool closing = false;
    bool level = false;       // <level>, only in patterns
    std::string_view name;    // empty for </>
    std::string_view code;
};

[[nodiscard]] constexpr bool parse_tag(std::string_view text, std::size_t pos, bool allow_level, Tag& tag) noexcept {
    const auto end = text.find('>', pos + 1);
    if (end == std::string_view::npos || end - pos > 16) return false;
    auto name = text.substr(pos + 1, end - pos - 1);
    tag = Tag{};
    tag.length = end - pos + 1;
    if (!name.empty() && name.front() == '/') {
        tag.closing = true;
        name.remove_prefix(1);
    }
    tag.name = name;
    if (tag.closing && name.empty()) return true;
    if (allow_level && name == "level") {
        tag.level = true;
        return true;
    }
    if (const auto* style = find_style(name)) {
        tag.code = style->code;
        return true;
    }
    return false;
}

// Length of the replacement field starting at fmt[pos] == '{', nested fields included (0 if unterminated).
[[nodiscard]] constexpr std::size_t field_length(std::string_view fmt, std::size_t pos) noexcept {
    std::size_t depth = 0;
    for (std::size_t i = pos; i < fmt.size(); ++i) {
        if (fmt[i] == '{') ++depth;
        else if (fmt[i] == '}' && --depth == 0) return i - pos + 1;
    }
    return 0;
}

// True when `text` holds a color tag. In a format string, replacement fields are skipped: "{:<8}" is no tag.
[[nodiscard]] constexpr bool has_tags(std::string_view text, bool format_string) noexcept {
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (format_string && (c == '{' || c == '}')) {
            if (i + 1 < text.size() && text[i + 1] == c) ++i;
            else if (c == '{') i += std::max<std::size_t>(field_length(text, i), 1) - 1;
            continue;
        }
        Tag tag;
        if (c == '<' && parse_tag(text, i, false, tag)) return true;
    }
    return false;
}

// Copies `text` with its color tags turned into ANSI codes (color) or removed (no color).
inline void render_tags(std::string_view text, bool format_string, bool color, std::string& out) {
    std::array<Tag, 16> open{};
    std::size_t depth = 0;
    for (std::size_t i = 0; i < text.size();) {
        const char c = text[i];
        if (format_string && (c == '{' || c == '}')) {
            std::size_t length = 1;
            if (i + 1 < text.size() && text[i + 1] == c) length = 2;
            else if (c == '{') length = std::max<std::size_t>(field_length(text, i), 1);
            out.append(text.substr(i, length));
            i += length;
            continue;
        }
        Tag tag;
        if (c != '<' || !parse_tag(text, i, false, tag)) {
            out += c;
            ++i;
            continue;
        }
        i += tag.length;
        if (!tag.closing) {
            if (depth < open.size()) open[depth++] = tag;
            if (color) out += tag.code;
            continue;
        }
        // Closing tag: drop the innermost matching style, then restore the ones still open.
        std::size_t k = depth;
        while (k > 0 && !tag.name.empty() && open[k - 1].name != tag.name) --k;
        if (k == 0) continue;
        for (std::size_t j = k - 1; j + 1 < depth; ++j) open[j] = open[j + 1];
        --depth;
        if (color) {
            out += ansi_reset;
            for (std::size_t j = 0; j < depth; ++j) out += open[j].code;
        }
    }
    if (color && depth > 0) out += ansi_reset;
}

// -- Time -----------------------------------------------------------------------------

inline void to_tm(std::time_t time, bool utc, std::tm& out) noexcept {
#if defined(_WIN32)
    if (utc) gmtime_s(&out, &time);
    else localtime_s(&out, &time);
#else
    if (utc) gmtime_r(&time, &out);
    else localtime_r(&time, &out);
#endif
}

struct CivilTime {
    char date[10];   // YYYY-MM-DD
    char time[12];   // HH:MM:SS.mmm
};

inline void put2(char* out, int value) noexcept {
    out[0] = static_cast<char>('0' + value / 10 % 10);
    out[1] = static_cast<char>('0' + value % 10);
}

// Local or UTC date and time of `tp`. The calendar conversion is cached per thread and per second.
[[nodiscard]] inline CivilTime civil_time(std::chrono::system_clock::time_point tp, bool utc) noexcept {
    struct Cache {
        std::int64_t second = (std::numeric_limits<std::int64_t>::min)();
        char date[10]{};
        char hms[8]{};
    };
    thread_local Cache caches[2];
    auto& cache = caches[utc ? 1 : 0];
    const auto floored = std::chrono::floor<std::chrono::seconds>(tp);
    const std::int64_t second = floored.time_since_epoch().count();
    if (cache.second != second) {
        std::tm tm{};
        to_tm(static_cast<std::time_t>(second), utc, tm);
        const int year = tm.tm_year + 1900;
        put2(cache.date, year / 100);
        put2(cache.date + 2, year % 100);
        cache.date[4] = '-';
        put2(cache.date + 5, tm.tm_mon + 1);
        cache.date[7] = '-';
        put2(cache.date + 8, tm.tm_mday);
        put2(cache.hms, tm.tm_hour);
        cache.hms[2] = ':';
        put2(cache.hms + 3, tm.tm_min);
        cache.hms[5] = ':';
        put2(cache.hms + 6, tm.tm_sec);
        cache.second = second;
    }
    CivilTime out;
    std::copy_n(cache.date, 10, out.date);
    std::copy_n(cache.hms, 8, out.time);
    const auto ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(tp - floored).count());
    out.time[8] = '.';
    out.time[9] = static_cast<char>('0' + ms / 100);
    put2(out.time + 10, ms % 100);
    return out;
}

// -- Text helpers ---------------------------------------------------------------------

// Length of the valid UTF-8 sequence starting at s[i], 0 if invalid.
[[nodiscard]] inline std::size_t utf8_length(std::string_view s, std::size_t i) noexcept {
    const auto byte = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
    const unsigned char lead = byte(i);
    std::size_t length = 0;
    unsigned char low = 0x80, high = 0xBF;
    if (lead >= 0xC2 && lead <= 0xDF) length = 2;
    else if (lead >= 0xE0 && lead <= 0xEF) {
        length = 3;
        if (lead == 0xE0) low = 0xA0;
        if (lead == 0xED) high = 0x9F;
    } else if (lead >= 0xF0 && lead <= 0xF4) {
        length = 4;
        if (lead == 0xF0) low = 0x90;
        if (lead == 0xF4) high = 0x8F;
    } else return 0;
    if (i + length > s.size() || byte(i + 1) < low || byte(i + 1) > high) return 0;
    for (std::size_t k = 2; k < length; ++k)
        if (byte(i + k) < 0x80 || byte(i + k) > 0xBF) return 0;
    return length;
}

// Escapes for a JSON string. Invalid UTF-8 becomes U+FFFD, so the output is always valid JSON.
inline void json_escape(std::string_view s, std::string& out) {
    constexpr char hex[] = "0123456789abcdef";
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        if (c >= 0x80) {
            const auto length = utf8_length(s, i);
            if (length == 0) {
                out += "\\ufffd";
                ++i;
            } else {
                out.append(s.substr(i, length));
                i += length;
            }
            continue;
        }
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        default:
            if (c < 0x20) {
                out += "\\u00";
                out += hex[c >> 4];
                out += hex[c & 0xF];
            } else {
                out += static_cast<char>(c);
            }
        }
        ++i;
    }
}

[[nodiscard]] constexpr std::string_view file_basename(std::string_view path) noexcept {
    const auto slash = path.find_last_of("/\\");
    return slash == std::string_view::npos ? path : path.substr(slash + 1);
}

// "int __cdecl Game::update(float)" -> "Game::update": return type, calling convention and parameters removed.
[[nodiscard]] constexpr std::string_view function_basename(std::string_view s) noexcept {
    constexpr auto is_ident = [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
    };
    std::size_t start = 0, depth = 0, i = 0;
    while (i < s.size()) {
        const char c = s[i];
        if (depth == 0 && c == 'o' && s.substr(i).starts_with("operator") && (i == 0 || !is_ident(s[i - 1]))
            && (i + 8 == s.size() || !is_ident(s[i + 8]))) {
            i += 8;   // operator(), operator<, operator new...: the name ends at the parameter list
            while (i < s.size() && s[i] == ' ') ++i;
            if (s.substr(i).starts_with("()")) i += 2;
            while (i < s.size() && s[i] != '(') ++i;
            break;
        }
        if (c == '<') ++depth;
        else if (c == '>' && depth > 0) --depth;
        else if (depth == 0 && c == '(') break;
        else if (depth == 0 && c == ' ') start = i + 1;
        ++i;
    }
    if (i >= s.size() || i <= start) return s;
    return s.substr(start, i - start);
}

// -- Terminal -------------------------------------------------------------------------

// File descriptor of a stream, -1 when there is none (a Windows GUI program has no console).
[[nodiscard]] inline int stream_fd(std::FILE* stream) noexcept {
    if (!stream) return -1;
#if defined(_WIN32)
    const int fd = _fileno(stream);
#else
    const int fd = ::fileno(stream);
#endif
    return fd < 0 ? -1 : fd;
}

[[nodiscard]] inline bool is_terminal(std::FILE* stream) noexcept {
    const int fd = stream_fd(stream);
#if defined(_WIN32)
    return fd >= 0 && _isatty(fd) != 0;
#else
    return fd >= 0 && ::isatty(fd) != 0;
#endif
}

// Windows consoles show ANSI codes only once asked to; other terminals always do.
inline bool enable_ansi(std::FILE* stream) noexcept {
#if defined(_WIN32)
    const int fd = stream_fd(stream);
    if (fd < 0) return false;
    auto* console = reinterpret_cast<void*>(_get_osfhandle(fd));
    unsigned long mode = 0;
    if (!GetConsoleMode(console, &mode)) return false;
    constexpr unsigned long processed_output = 0x0001, virtual_terminal = 0x0004;
    return (mode & virtual_terminal) != 0 || SetConsoleMode(console, mode | processed_output | virtual_terminal) != 0;
#else
    (void)stream;
    return true;
#endif
}

[[nodiscard]] inline std::string environment(const char* name) {
#if defined(_MSC_VER)
    char* value = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&value, &size, name) != 0 || !value) return {};
    std::string text(value);
    std::free(value);
    return text;
#else
    const char* value = std::getenv(name);
    return value ? value : "";
#endif
}

// Colors when the stream is a terminal, unless NO_COLOR is set; FORCE_COLOR or CLICOLOR_FORCE force them.
[[nodiscard]] inline bool colors_wanted(std::FILE* stream) {
    if (!environment("NO_COLOR").empty()) return false;
    auto force = environment("FORCE_COLOR");
    if (force.empty()) force = environment("CLICOLOR_FORCE");
    if (!force.empty() && force != "0") {
        enable_ansi(stream);
        return true;
    }
    if (!is_terminal(stream)) return false;
#if !defined(_WIN32)
    if (environment("TERM") == "dumb") return false;
#endif
    return enable_ansi(stream);
}

// std::print writes UTF-8 to a Windows console through the Unicode API (when compiled with /utf-8).
inline void write_console(std::FILE* stream, std::string_view text) {
#if defined(__cpp_lib_print)
    std::print(stream, "{}", text);
#else
    std::fwrite(text.data(), 1, text.size(), stream);
#endif
}

// The logger's own problems go to stderr, at most 20 times in a run.
inline void internal_error(std::string_view what, std::string_view detail = {}) noexcept {
    static std::atomic<int> count{0};
    const int n = count.fetch_add(1, std::memory_order_relaxed);
    if (n < 20) {
        std::fprintf(stderr, "[Gem::Log] %.*s%.*s\n", static_cast<int>(what.size()), what.data(),
                     static_cast<int>(detail.size()), detail.data());
    } else if (n == 20) {
        std::fputs("[Gem::Log] further errors are not shown\n", stderr);
    }
}

[[nodiscard]] inline std::string thread_id_label() {
#if defined(__linux__)
    return std::to_string(static_cast<long long>(::syscall(SYS_gettid)));
#elif defined(__cpp_lib_formatters)
    return std::format("{}", std::this_thread::get_id());
#else
    std::ostringstream text;
    text << std::this_thread::get_id();
    return text.str();
#endif
}

[[nodiscard]] inline std::string& thread_label() {
    thread_local std::string label = thread_id_label();
    return label;
}

[[nodiscard]] inline std::string path_text(const std::filesystem::path& path) {
    const auto text = path.u8string();
    return std::string(reinterpret_cast<const char*>(text.data()), text.size());
}

// -- Values ---------------------------------------------------------------------------

template <class>
inline constexpr bool always_false = false;

template <class T>
struct Streamed {
    const T& value;
};

template <class T>
concept Streamable = requires(std::ostream& out, const T& value) { out << value; };

#if defined(__cpp_lib_format_ranges)
template <class T>
concept StdFormattable = std::formattable<T, char>;
#else
template <class T>
concept StdFormattable = std::is_default_constructible_v<std::formatter<T, char>>;
#endif

template <class T>
concept CharPointer = std::is_pointer_v<T> && std::is_same_v<std::remove_cv_t<std::remove_pointer_t<T>>, char>;

// Makes any value printable by std::format. Types with a std::formatter pass through unchanged, the others
// get a fallback: exception -> what(), path -> UTF-8 text, enum -> value, pointer -> address, operator<<.
template <class T>
[[nodiscard]] decltype(auto) wrap(const T& value) {
    if constexpr (std::is_base_of_v<std::exception, T>) return std::string_view(value.what());
    else if constexpr (std::is_same_v<T, std::filesystem::path>) return path_text(value);
    else if constexpr (CharPointer<T>) return std::string_view(value ? value : "(null)");
    else if constexpr (StdFormattable<T>) return (value);
    else if constexpr (std::is_enum_v<T>) return std::to_underlying(value);
    else if constexpr (std::is_pointer_v<T> && std::is_object_v<std::remove_pointer_t<T>>) return static_cast<const void*>(value);
    else if constexpr (Streamable<T>) return Streamed<T>{value};
    else static_assert(always_false<T>, "Gem::Log: this type cannot be formatted, give it a std::formatter or an operator<<");
}

template <class T>
using wrapped_t = std::remove_cvref_t<decltype(wrap(std::declval<const std::remove_cvref_t<T>&>()))>;

} // namespace detail
} // namespace Gem::Log

namespace std {
template <class T>
struct formatter<Gem::Log::detail::Streamed<T>, char> : formatter<string_view, char> {
    template <class FormatContext>
    auto format(const Gem::Log::detail::Streamed<T>& streamed, FormatContext& context) const {
        ostringstream text;
        text << streamed.value;
        return formatter<string_view, char>::format(text.str(), context);
    }
};
} // namespace std

namespace Gem::Log {

// Format string known only at run time: Gem::Log::info(Gem::Log::runtime_format(text), args...).
struct RuntimeFormat {
    std::string_view text;
};

[[nodiscard]] constexpr RuntimeFormat runtime_format(std::string_view text) noexcept { return {text}; }

namespace detail {

// Format string of a log or a print: checked at compile time against its arguments, and remembers the
// file, line and function of the call.
template <class... Args>
struct FormatString {
    std::string_view text;
    std::source_location location;
    bool tags = false;

    template <class S>
        requires std::is_convertible_v<const S&, std::string_view>
    consteval FormatString(const S& format, std::source_location loc = std::source_location::current())
        : text(format), location(loc), tags(has_tags(text, true)) {
        (void)std::format_string<Args...>(format);
    }

    FormatString(RuntimeFormat format, std::source_location loc = std::source_location::current()) noexcept
        : text(format.text), location(loc), tags(has_tags(text, true)) {}
};

template <class... Args>
using Format = FormatString<wrapped_t<Args>...>;

template <class... Args>
void format_into(std::string& out, std::string_view format, const Args&... args) {
    [&](const auto&... values) {
        std::vformat_to(std::back_inserter(out), format, std::make_format_args(values...));
    }(wrap(args)...);
}

// Text and JSON kind of a value, for Gem::Log::Context and LOG_DUMP.
template <class T>
void field_value(const T& value, std::string& out, FieldKind& kind) {
    out.clear();
    if constexpr (std::is_same_v<T, bool>) {
        out = value ? "true" : "false";
        kind = FieldKind::Bool;
    } else if constexpr (std::is_same_v<T, char>) {
        out.assign(1, value);
        kind = FieldKind::String;
    } else if constexpr (std::is_floating_point_v<T>) {
        out = std::format("{}", value);
        kind = std::isfinite(value) ? FieldKind::Number : FieldKind::Other;   // JSON has no nan or inf
    } else if constexpr (std::is_integral_v<T> && StdFormattable<T>) {
        out = std::format("{}", value);
        kind = FieldKind::Number;
    } else if constexpr (std::is_enum_v<T> && !StdFormattable<T>) {
        out = std::format("{}", std::to_underlying(value));
        kind = FieldKind::Number;
    } else {
        format_into(out, "{}", value);
        constexpr bool text = CharPointer<T> || std::is_convertible_v<const T&, std::string_view>
                              || std::is_same_v<T, std::filesystem::path>;
        kind = text ? FieldKind::String : FieldKind::Other;
    }
}

inline void append_quoted(std::string& out, std::string_view text) {
#if defined(__cpp_lib_format_ranges)
    std::format_to(std::back_inserter(out), "{:?}", text);
#else
    out += '"';
    out += text;
    out += '"';
#endif
}

// Splits the text of LOG_DUMP's arguments ("a, f(b, c), v[0]") into one expression per argument.
template <std::size_t N>
[[nodiscard]] std::array<std::string_view, N> split_names(std::string_view text) {
    constexpr auto trim = [](std::string_view s) {
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\n')) s.remove_prefix(1);
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\n')) s.remove_suffix(1);
        return s;
    };
    std::array<std::string_view, N> names{};
    for (const bool angles : {false, true}) {   // template arguments "pair<int, int>" are tried second
        std::size_t count = 0, depth = 0, begin = 0;
        char quote = 0;
        for (std::size_t i = 0; i <= text.size(); ++i) {
            const char c = i < text.size() ? text[i] : ',';
            if (quote != 0) {
                if (c == '\\') ++i;
                else if (c == quote) quote = 0;
                continue;
            }
            if (c == '"' || c == '\'') quote = c;
            else if (c == '(' || c == '[' || c == '{' || (angles && c == '<')) ++depth;
            else if ((c == ')' || c == ']' || c == '}' || (angles && c == '>')) && depth > 0) --depth;
            else if (c == ',' && depth == 0) {
                if (count < N) names[count] = trim(text.substr(begin, i - begin));
                ++count;
                begin = i + 1;
            }
        }
        if (count == N) return names;
    }
    names.fill("?");
    return names;
}

struct ChannelData;
[[nodiscard]] inline bool enabled(Level level, const ChannelData* channel) noexcept;
inline void submit(Level level, const ChannelData* channel, const std::source_location& location,
                   std::string_view message, std::string_view ansi_message, std::span<const Field> vars) noexcept;

template <class... Args>
void dump(const std::source_location& location, std::string_view names, const Args&... args) {
    if (!enabled(Level::Debug, nullptr)) return;
    constexpr std::size_t count = sizeof...(Args);
    const auto keys = split_names<count>(names);
    std::array<std::string, count> values;
    std::array<FieldKind, count> kinds{};
    std::array<Field, count> fields;
    std::string message;
    try {
        std::size_t i = 0;
        ((field_value(args, values[i], kinds[i]), ++i), ...);
        for (std::size_t k = 0; k < count; ++k) {
            fields[k] = Field{keys[k], values[k], kinds[k]};
            if (k > 0) message += ", ";
            message += keys[k];
            message += " = ";
            if (kinds[k] == FieldKind::String) append_quoted(message, values[k]);
            else message += values[k];
        }
    } catch (const std::exception& e) {
        message = std::format("{} [format error: {}]", names, e.what());
        submit(Level::Debug, nullptr, location, message, message, {});
        return;
    }
    submit(Level::Debug, nullptr, location, message, message, fields);
}

// -- Patterns -------------------------------------------------------------------------

using LevelStyles = std::array<std::string, 7>;

struct StyleRef {
    bool level = false;      // color of the record's level
    std::string_view code;   // otherwise a fixed ANSI code
};

enum class Token : std::uint8_t {
    Literal, Push, Pop, GroupBegin, GroupEnd,
    Level, Message, Date, Time, Elapsed, Thread, File, Path, Line, Function, Channel, Context, ContextKey
};

struct Segment {
    Token token = Token::Literal;
    std::string text;               // literal text, or the key of %(context[key])
    std::string format;             // "{:<8}" when the token has a format spec
    StyleRef style;                 // Push
    std::vector<StyleRef> restore;  // Pop: styles still open, applied again after the reset
};

[[nodiscard]] inline std::string_view find_field(std::span<const Field> fields, std::string_view key) noexcept {
    for (auto it = fields.rbegin(); it != fields.rend(); ++it)
        if (it->key == key) return it->value;
    return {};
}

// A field hidden by a later one with the same key (nested Context).
[[nodiscard]] inline bool shadowed(std::span<const Field> fields, std::size_t index) noexcept {
    for (auto i = index + 1; i < fields.size(); ++i)
        if (fields[i].key == fields[index].key) return true;
    return false;
}

inline void render_context(std::span<const Field> fields, std::string& out) {
    for (std::size_t i = 0; i < fields.size(); ++i) {
        if (shadowed(fields, i)) continue;
        if (!out.empty()) out += ' ';
        out += fields[i].key;
        out += '=';
        out += fields[i].value;
    }
}

[[noreturn]] inline void bad_pattern(std::string_view pattern, std::string_view why) {
    throw std::invalid_argument(std::format("Gem::Log: invalid pattern \"{}\": {}", pattern, why));
}

// A pattern compiled once: tokens, format specs and color tags are resolved before the first record.
class Pattern {
public:
    Pattern() = default;
    explicit Pattern(std::string_view pattern);

    void render(const Record& record, bool color, const LevelStyles& level_styles,
                std::chrono::system_clock::time_point start, std::string& out) const;

private:
    void add_token(std::string_view pattern, std::string_view body);

    std::vector<Segment> segments_;
};

inline Pattern::Pattern(std::string_view pattern) {
    struct Open {
        std::string_view name;
        StyleRef style;
    };
    constexpr auto none = std::string_view::npos;
    std::vector<Open> open;
    std::size_t group_depth = none;   // tags open when %[ started
    const auto literal = [&](std::string_view text) {
        if (segments_.empty() || segments_.back().token != Token::Literal) segments_.emplace_back();
        segments_.back().text += text;
    };
    for (std::size_t i = 0; i < pattern.size();) {
        const char c = pattern[i];
        const char next = i + 1 < pattern.size() ? pattern[i + 1] : '\0';
        if (c == '%' && next == '%') {
            literal("%");
            i += 2;
        } else if (c == '%' && next == '[') {
            if (group_depth != none) bad_pattern(pattern, "%[ groups cannot be nested");
            group_depth = open.size();
            segments_.push_back({.token = Token::GroupBegin});
            i += 2;
        } else if (c == '%' && next == ']') {
            if (group_depth == none) bad_pattern(pattern, "%] without %[");
            if (open.size() != group_depth) bad_pattern(pattern, "color tags must be closed inside their %[ %] group");
            group_depth = none;
            segments_.push_back({.token = Token::GroupEnd});
            i += 2;
        } else if (c == '%' && next == '(') {
            const auto close = pattern.find(')', i + 2);
            if (close == none) bad_pattern(pattern, "missing ')'");
            add_token(pattern, pattern.substr(i + 2, close - i - 2));
            i = close + 1;
        } else if (Tag tag; c == '<' && parse_tag(pattern, i, true, tag)) {
            i += tag.length;
            if (!tag.closing) {
                const StyleRef style{tag.level, tag.code};
                open.push_back({tag.name, style});
                segments_.push_back({.token = Token::Push, .style = style});
                continue;
            }
            auto k = open.size();
            while (k > 0 && !tag.name.empty() && open[k - 1].name != tag.name) --k;
            if (k == 0) continue;   // nothing to close
            open.erase(open.begin() + static_cast<std::ptrdiff_t>(k - 1));
            Segment pop{.token = Token::Pop};
            for (const auto& still_open : open) pop.restore.push_back(still_open.style);
            segments_.push_back(std::move(pop));
        } else {
            literal(pattern.substr(i, 1));
            ++i;
        }
    }
    if (group_depth != none) bad_pattern(pattern, "missing %]");
    if (!open.empty()) segments_.push_back({.token = Token::Pop});
}

inline void Pattern::add_token(std::string_view pattern, std::string_view body) {
    static constexpr std::pair<std::string_view, Token> names[] = {
        {"level", Token::Level},     {"levelname", Token::Level}, {"message", Token::Message}, {"msg", Token::Message},
        {"date", Token::Date},       {"time", Token::Time},       {"elapsed", Token::Elapsed}, {"thread", Token::Thread},
        {"file", Token::File},       {"path", Token::Path},       {"line", Token::Line},       {"function", Token::Function},
        {"func", Token::Function},   {"channel", Token::Channel}, {"name", Token::Channel},    {"context", Token::Context}};
    const auto bracket = body.find(']');
    const auto colon = body.find(':', bracket == std::string_view::npos ? 0 : bracket);
    const auto name = body.substr(0, colon);
    const auto spec = colon == std::string_view::npos ? std::string_view() : body.substr(colon + 1);
    Segment segment;
    if (name.starts_with("context[") && name.ends_with("]")) {
        segment.token = Token::ContextKey;
        segment.text = name.substr(8, name.size() - 9);
    } else {
        const auto* found = std::ranges::find(names, name, &std::pair<std::string_view, Token>::first);
        if (found == std::end(names)) bad_pattern(pattern, std::format("unknown token %({})", name));
        segment.token = found->second;
    }
    if (!spec.empty() || segment.token == Token::Elapsed) {
        segment.format = std::format("{{:{}}}", spec.empty() ? std::string_view(".3f") : spec);
        try {   // a bad spec fails here, not at every record
            std::string probe;
            if (segment.token == Token::Line) {
                unsigned value = 1;
                std::vformat_to(std::back_inserter(probe), segment.format, std::make_format_args(value));
            } else if (segment.token == Token::Elapsed) {
                double value = 1;
                std::vformat_to(std::back_inserter(probe), segment.format, std::make_format_args(value));
            } else {
                std::string_view value = "x";
                std::vformat_to(std::back_inserter(probe), segment.format, std::make_format_args(value));
            }
        } catch (const std::format_error& error) {
            bad_pattern(pattern, std::format("bad format spec in %({}): {}", body, error.what()));
        }
    }
    segments_.push_back(std::move(segment));
}

inline void Pattern::render(const Record& record, bool color, const LevelStyles& level_styles,
                            std::chrono::system_clock::time_point start, std::string& out) const {
    const auto code = [&](const StyleRef& style) -> std::string_view {
        if (!style.level) return style.code;
        const auto index = static_cast<std::size_t>(record.level);
        return index < level_styles.size() ? std::string_view(level_styles[index]) : std::string_view();
    };
    constexpr auto none = std::string::npos;
    std::size_t group = none;
    bool group_ok = true;
    std::optional<CivilTime> civil;
    std::string context;
    for (const auto& segment : segments_) {
        std::string_view text;
        bool is_number = false, is_seconds = false;
        unsigned number = 0;
        double seconds = 0;
        switch (segment.token) {
        case Token::Literal: out += segment.text; continue;
        case Token::Push:
            if (color) out += code(segment.style);
            continue;
        case Token::Pop:
            if (color) {
                out += ansi_reset;
                for (const auto& style : segment.restore) out += code(style);
            }
            continue;
        case Token::GroupBegin:
            group = out.size();
            group_ok = true;
            continue;
        case Token::GroupEnd:
            if (!group_ok) out.resize(group);
            group = none;
            continue;
        case Token::Level: text = to_string(record.level); break;
        case Token::Message: text = color && !record.ansi_message.empty() ? record.ansi_message : record.message; break;
        case Token::Date:
        case Token::Time:
            if (!civil) civil = civil_time(record.time, false);
            text = segment.token == Token::Date ? std::string_view(civil->date, 10) : std::string_view(civil->time, 12);
            break;
        case Token::Elapsed:
            seconds = std::chrono::duration<double>(record.time - start).count();
            is_seconds = true;
            break;
        case Token::Thread: text = record.thread; break;
        case Token::File: text = file_basename(record.location.file_name()); break;
        case Token::Path: text = record.location.file_name(); break;
        case Token::Line:
            number = record.location.line();
            is_number = true;
            break;
        case Token::Function: text = function_basename(record.location.function_name()); break;
        case Token::Channel: text = record.channel; break;
        case Token::Context:
            context.clear();
            render_context(record.context, context);
            text = context;
            break;
        case Token::ContextKey: text = find_field(record.context, segment.text); break;
        }
        if (!is_number && !is_seconds && text.empty() && group != none) group_ok = false;
        if (segment.format.empty()) {
            if (is_number) std::format_to(std::back_inserter(out), "{}", number);
            else out += text;
        } else if (is_number) {
            std::vformat_to(std::back_inserter(out), segment.format, std::make_format_args(number));
        } else if (is_seconds) {
            std::vformat_to(std::back_inserter(out), segment.format, std::make_format_args(seconds));
        } else {
            std::vformat_to(std::back_inserter(out), segment.format, std::make_format_args(text));
        }
    }
}

// One JSON object per record (JSON Lines), time in UTC.
inline void render_json(const Record& record, std::string& out) {
    const auto civil = civil_time(record.time, true);
    out += R"({"time":")";
    out.append(civil.date, 10);
    out += 'T';
    out.append(civil.time, 12);
    out += R"(Z","level":")";
    out += to_string(record.level);
    out += R"(","message":")";
    json_escape(record.message, out);
    out += R"(","thread":")";
    json_escape(record.thread, out);
    out += R"(","file":")";
    json_escape(record.location.file_name(), out);
    out += R"(","line":)";
    std::format_to(std::back_inserter(out), "{}", record.location.line());
    out += R"(,"function":")";
    json_escape(function_basename(record.location.function_name()), out);
    out += '"';
    if (!record.channel.empty()) {
        out += R"(,"channel":")";
        json_escape(record.channel, out);
        out += '"';
    }
    const auto object = [&](std::string_view name, std::span<const Field> fields) {
        if (fields.empty()) return;
        out += ",\"";
        out += name;
        out += "\":{";
        bool first = true;
        for (std::size_t i = 0; i < fields.size(); ++i) {
            if (shadowed(fields, i)) continue;
            if (!first) out += ',';
            first = false;
            out += '"';
            json_escape(fields[i].key, out);
            out += "\":";
            if (fields[i].kind == FieldKind::Number || fields[i].kind == FieldKind::Bool) {
                out += fields[i].value;
            } else {
                out += '"';
                json_escape(fields[i].value, out);
                out += '"';
            }
        }
        out += '}';
    };
    object("context", record.context);
    object("vars", record.vars);
    out += '}';
}

struct Core;
[[nodiscard]] inline Core& core();

} // namespace detail

// -- Sinks ----------------------------------------------------------------------------

// Base of every output. A custom sink derives from it and implements write():
//     struct Overlay : Gem::Log::Sink {
//         void write(const Gem::Log::Record& record, std::string_view line) override { ... }
//     };
//     Gem::Log::add_sink(std::make_shared<Overlay>());
class Sink {
public:
    Sink() : Sink(default_file_pattern) {}
    virtual ~Sink() = default;
    Sink(const Sink&) = delete;
    Sink& operator=(const Sink&) = delete;

    void set_level(Level minimum);
    [[nodiscard]] Level level() const noexcept { return level_.load(std::memory_order_relaxed); }
    void set_pattern(std::string_view pattern);                    // throws std::invalid_argument
    void set_json(bool enabled = true);                            // JSON Lines instead of the pattern
    void set_filter(std::function<bool(const Record&)> filter);   // returning false drops the record

protected:
    explicit Sink(std::string_view pattern, bool json = false) : pattern_(pattern), json_(json) {}

    // Receives every record this sink accepts, `line` ending with '\n'. Always called under the logger's
    // lock, so a sink needs no lock of its own.
    virtual void write(const Record& record, std::string_view line) = 0;
    virtual void flush() {}
    [[nodiscard]] virtual bool colored() const noexcept { return false; }

private:
    friend struct detail::Core;

    [[nodiscard]] virtual const std::filesystem::path* file() const noexcept { return nullptr; }
    bool handle(const Record& record) noexcept;

    std::atomic<Level> level_{Level::Trace};
    detail::Pattern pattern_;
    bool json_ = false;
    std::function<bool(const Record&)> filter_;
    std::string line_;
};

// stdout (or stderr), colored when it is a terminal. One is created at start-up: Gem::Log::console().
class ConsoleSink final : public Sink {
public:
    explicit ConsoleSink(std::FILE* stream = stdout, ColorMode mode = ColorMode::Auto)
        : Sink(default_console_pattern), stream_(stream), mode_(mode), color_(colors_for(stream, mode)) {}

    void set_color(ColorMode mode);
    void set_stream(std::FILE* stream);
    [[nodiscard]] bool color_enabled() const noexcept { return color_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::FILE* stream() const noexcept { return stream_.load(std::memory_order_relaxed); }

protected:
    void write(const Record&, std::string_view line) override {
        if (auto* out = stream(); detail::stream_fd(out) >= 0) detail::write_console(out, line);
    }
    void flush() override {
        if (auto* out = stream(); detail::stream_fd(out) >= 0) std::fflush(out);
    }
    [[nodiscard]] bool colored() const noexcept override { return color_enabled(); }

private:
    [[nodiscard]] static bool colors_for(std::FILE* stream, ColorMode mode) {
        if (mode == ColorMode::Never) return false;
        if (mode == ColorMode::Always) {
            detail::enable_ansi(stream);
            return true;
        }
        return detail::colors_wanted(stream);
    }

    std::atomic<std::FILE*> stream_;
    ColorMode mode_;
    std::atomic<bool> color_;
};

// A log file, rotated by size: app.log -> app.1.log -> app.2.log ... Directories are created when needed.
class FileSink final : public Sink {
public:
    explicit FileSink(std::filesystem::path path, FileOptions options = {})
        : Sink(default_file_pattern, options.json), path_(std::move(path)), options_(options) {
        open(options.truncate);
    }
    ~FileSink() override {
        if (file_) std::fclose(file_);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] bool is_open() const noexcept { return open_.load(std::memory_order_relaxed); }

protected:
    void write(const Record& record, std::string_view line) override;
    void flush() override;

private:
    [[nodiscard]] const std::filesystem::path* file() const noexcept override { return &path_; }
    bool open(bool truncate);
    void rotate();
    [[nodiscard]] std::filesystem::path rotated(std::size_t index) const;
    void fail(std::string_view what, const std::error_code& error);

    std::filesystem::path path_;
    FileOptions options_;
    std::FILE* file_ = nullptr;
    std::atomic<bool> open_{false};
    std::uint64_t size_ = 0;
    std::chrono::steady_clock::time_point retry_{};
    bool failing_ = false;   // an error was reported: stay quiet until it works again
};

// Calls a function with each record and its line (without '\n'): in-game consoles, IDE output, tests...
class CallbackSink final : public Sink {
public:
    using Callback = std::function<void(const Record& record, std::string_view line)>;

    explicit CallbackSink(Callback callback) : Sink(default_console_pattern), callback_(std::move(callback)) {}

protected:
    void write(const Record& record, std::string_view line) override {
        if (!line.empty() && line.back() == '\n') line.remove_suffix(1);
        if (callback_) callback_(record, line);
    }

private:
    Callback callback_;
};

// -- Core -----------------------------------------------------------------------------

namespace detail {

struct ChannelData {
    explicit ChannelData(std::string channel_name) : name(std::move(channel_name)) {}
    const std::string name;
    std::atomic<Level> level{Level::Trace};
};

struct OwnedField {
    std::string key, value;
    FieldKind kind = FieldKind::String;
};

// Gem::Log::Context values of the current thread.
struct ThreadContext {
    struct Entry {
        std::uint64_t id = 0;
        OwnedField field;
    };
    std::vector<Entry> entries;
    std::vector<Field> views;
    std::uint64_t next_id = 0;

    void rebuild() {
        views.clear();
        for (const auto& entry : entries) views.push_back({entry.field.key, entry.field.value, entry.field.kind});
    }
};

[[nodiscard]] inline ThreadContext& thread_context() {
    thread_local ThreadContext context;
    return context;
}

// A record copied into the async queue.
struct QueuedRecord {
    Level level = Level::Info;
    std::chrono::system_clock::time_point time{};
    std::source_location location{};
    std::string_view channel;   // channel names live until the end of the program
    std::string message, ansi_message, thread;
    std::vector<OwnedField> context, vars;
};

// Above 0 while this thread holds the logger's lock, that is while it runs sinks. A log from there is dropped
// instead of dead-locking; configuring from there works.
inline thread_local int lock_depth = 0;

struct LockDepth {
    LockDepth() noexcept { ++lock_depth; }
    ~LockDepth() { --lock_depth; }
    LockDepth(const LockDepth&) = delete;
    LockDepth& operator=(const LockDepth&) = delete;
};

[[nodiscard]] inline std::filesystem::path normalized(const std::filesystem::path& path) {
    std::error_code error;
    const auto absolute = std::filesystem::absolute(path, error);
    if (error) return path.lexically_normal();
    auto canonical = std::filesystem::weakly_canonical(absolute, error);
    return error ? absolute.lexically_normal() : canonical;
}

struct Core {
    Core();

    // Read on every log, without lock
    std::atomic<Level> level{Level::Trace};
    std::atomic<Level> sinks_level{Level::Trace};   // lowest level of the sinks
    std::atomic<bool> any_color{false};
    const std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    std::atomic<std::uint64_t> logged{0}, dropped{0};

    // Sinks and styles, guarded by `mutex`
    std::mutex mutex;
    std::vector<std::shared_ptr<Sink>> sinks;
    std::shared_ptr<ConsoleSink> console;
    LevelStyles level_styles{"\x1b[90m", "\x1b[36m", "\x1b[1m", "\x1b[1;32m", "\x1b[1;33m", "\x1b[1;31m", "\x1b[1;37;41m"};

    std::mutex channels_mutex;
    std::map<std::string, std::unique_ptr<ChannelData>, std::less<>> channels;

    // Async mode
    std::mutex async_mutex;   // serializes start_async() and stop_async()
    std::mutex queue_mutex;   // guards the members below
    std::condition_variable queue_not_empty, queue_not_full, queue_progress;
    std::deque<QueuedRecord> queue;
    AsyncOptions options;
    std::uint64_t pushed = 0, completed = 0;
    bool stopping = false, worker_done = true;
    std::atomic<bool> async{false};
    std::atomic<std::thread::id> worker_id{};

    template <class F>
    void locked(F&& f) {
        if (lock_depth > 0) {   // this thread already holds the lock
            f();
            return;
        }
        std::lock_guard lock(mutex);
        LockDepth depth;
        f();
    }

    void refresh() noexcept;
    void dispatch(const Record& record, bool flush_each) noexcept;
    void flush_sinks() noexcept;
    void write(const Record& record);
    void enqueue(const Record& record);
    void worker_loop() noexcept;
    void worker_run();
    void wait_for(std::uint64_t ticket);
    void drain();
    void start_async(AsyncOptions async_options);
    void stop_async();
    void flush();
    void print(std::string_view text);
    [[nodiscard]] bool print_colors();
    void set_pattern(const Pattern& pattern);
    [[nodiscard]] std::shared_ptr<FileSink> add_file(const std::filesystem::path& path, FileOptions file_options);
    [[nodiscard]] ChannelData* channel(std::string_view name);
    void shutdown() noexcept;
};

// Created on first use and never destroyed, so that logging still works in static destructors.
// Queued records are written by an atexit handler.
[[nodiscard]] inline Core& core() {
    static Core* const instance = [] {
        auto* created = new Core();
        std::atexit([] { core().shutdown(); });
        return created;
    }();
    return *instance;
}

inline Core::Core() {
    console = std::make_shared<ConsoleSink>(stdout);
    sinks.push_back(console);
    refresh();
}

inline void Core::refresh() noexcept {
    auto lowest = Level::Off;
    bool color = false;
    for (const auto& sink : sinks) {
        lowest = (std::min)(lowest, sink->level());
        color = color || (!sink->json_ && sink->colored());
    }
    sinks_level.store(lowest, std::memory_order_relaxed);
    any_color.store(color, std::memory_order_relaxed);
}

inline void Core::dispatch(const Record& record, bool flush_each) noexcept {
    logged.fetch_add(1, std::memory_order_relaxed);
    for (std::size_t i = 0; i < sinks.size(); ++i) {
        const auto sink = sinks[i];   // kept alive even if the sink removes itself
        if (sink->handle(record) && flush_each) {
            try {
                sink->flush();
            } catch (...) {
            }
        }
    }
}

inline void Core::flush_sinks() noexcept {
    for (std::size_t i = 0; i < sinks.size(); ++i) {
        try {
            sinks[i]->flush();
        } catch (...) {
        }
    }
}

// Synchronous mode: the calling thread writes the record, flushed at once.
inline void Core::write(const Record& record) {
    std::lock_guard lock(mutex);
    LockDepth depth;
    dispatch(record, true);
}

inline void Core::enqueue(const Record& record) {
    QueuedRecord item;
    item.level = record.level;
    item.time = record.time;
    item.location = record.location;
    item.channel = record.channel;
    item.message = record.message;
    if (record.ansi_message.data() != record.message.data()) item.ansi_message = record.ansi_message;
    item.thread = record.thread;
    for (const auto& field : record.context) item.context.push_back({std::string(field.key), std::string(field.value), field.kind});
    for (const auto& field : record.vars) item.vars.push_back({std::string(field.key), std::string(field.value), field.kind});

    std::uint64_t ticket = 0;
    bool wait = false;
    {
        std::unique_lock lock(queue_mutex);
        if (!stopping && !worker_done && queue.size() >= options.capacity) {
            switch (options.overflow) {
            case Overflow::Block:
                queue_not_full.wait(lock, [&] { return queue.size() < options.capacity || stopping || worker_done; });
                break;
            case Overflow::DropNewest:
                dropped.fetch_add(1, std::memory_order_relaxed);
                return;
            case Overflow::DropOldest:
                queue.pop_front();
                ++completed;
                dropped.fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
        if (!stopping && !worker_done) {
            queue.push_back(std::move(item));
            ticket = ++pushed;
            wait = record.level >= options.flush_level;
        }
    }
    if (ticket == 0) {   // async mode is stopping: write it here
        write(record);
        return;
    }
    queue_not_empty.notify_one();
    if (wait) wait_for(ticket);
}

inline void Core::worker_loop() noexcept {
    try {
        worker_run();
    } catch (...) {
        internal_error("the async writer stopped after an error");
    }
    std::lock_guard lock(queue_mutex);
    worker_done = true;
    queue_progress.notify_all();
    queue_not_full.notify_all();
}

inline void Core::worker_run() {
    std::deque<QueuedRecord> batch;
    std::vector<Field> context, vars;
    std::unique_lock lock(queue_mutex);
    for (;;) {
        queue_not_empty.wait(lock, [&] { return !queue.empty() || stopping; });
        if (queue.empty()) return;   // stopping, and everything is written
        batch.swap(queue);
        lock.unlock();
        queue_not_full.notify_all();
        {
            std::lock_guard sinks_lock(mutex);
            LockDepth depth;
            for (const auto& item : batch) {
                context.clear();
                vars.clear();
                for (const auto& field : item.context) context.push_back({field.key, field.value, field.kind});
                for (const auto& field : item.vars) vars.push_back({field.key, field.value, field.kind});
                Record record;
                record.level = item.level;
                record.time = item.time;
                record.message = item.message;
                record.ansi_message = item.ansi_message.empty() ? std::string_view(item.message) : item.ansi_message;
                record.channel = item.channel;
                record.thread = item.thread;
                record.location = item.location;
                record.context = context;
                record.vars = vars;
                dispatch(record, false);
            }
            flush_sinks();   // once per batch
        }
        const auto done = batch.size();
        batch.clear();
        lock.lock();
        completed += done;
        queue_progress.notify_all();
    }
}

inline void Core::wait_for(std::uint64_t ticket) {
    std::unique_lock lock(queue_mutex);
    queue_progress.wait(lock, [&] { return completed >= ticket || worker_done; });
}

// Waits until the records queued so far are written.
inline void Core::drain() {
    if (lock_depth > 0 || !async.load(std::memory_order_acquire) || std::this_thread::get_id() == worker_id.load()) return;
    std::uint64_t ticket = 0;
    {
        std::lock_guard lock(queue_mutex);
        ticket = pushed;
    }
    queue_not_empty.notify_one();
    wait_for(ticket);
}

inline void Core::start_async(AsyncOptions async_options) {
    async_options.capacity = std::max<std::size_t>(async_options.capacity, 1);
    std::lock_guard guard(async_mutex);
    {
        std::lock_guard lock(queue_mutex);
        options = async_options;
        if (async.load()) return;   // already running: the new options apply
        stopping = false;
        worker_done = false;
    }
    try {
        std::thread worker([this] { worker_loop(); });
        worker_id.store(worker.get_id());
        worker.detach();   // never joined: stop_async() waits for worker_done instead, which is safe at exit
    } catch (...) {
        std::lock_guard lock(queue_mutex);
        worker_done = true;
        throw;
    }
    async.store(true, std::memory_order_release);
}

inline void Core::stop_async() {
    if (lock_depth > 0 || std::this_thread::get_id() == worker_id.load()) return;   // not from a sink
    std::lock_guard guard(async_mutex);
    if (!async.load()) return;
    {
        std::lock_guard lock(queue_mutex);
        stopping = true;
    }
    queue_not_empty.notify_all();
    queue_not_full.notify_all();
    {
        std::unique_lock lock(queue_mutex);
        queue_progress.wait(lock, [&] { return worker_done; });
    }
    async.store(false, std::memory_order_release);
    worker_id.store(std::thread::id());
}

inline void Core::flush() {
    drain();
    locked([&] { flush_sinks(); });
}

inline void Core::print(std::string_view text) {
    drain();   // queued logs first: the output keeps its order
    locked([&] {
        if (stream_fd(stdout) < 0) return;
        try {
            write_console(stdout, text);
            std::fflush(stdout);
        } catch (...) {
        }
    });
}

inline bool Core::print_colors() {
    bool color = false;
    locked([&] {
        if (console && console->stream() == stdout) {
            color = console->color_enabled();
        } else {
            static const bool wanted = colors_wanted(stdout);
            color = wanted;
        }
    });
    return color;
}

inline void Core::set_pattern(const Pattern& pattern) {
    locked([&] {
        for (const auto& sink : sinks)
            if (!sink->json_) sink->pattern_ = pattern;
    });
}

inline std::shared_ptr<FileSink> Core::add_file(const std::filesystem::path& path, FileOptions file_options) {
    const auto key = normalized(path);
    std::shared_ptr<FileSink> result;
    locked([&] {
        for (const auto& sink : sinks) {
            if (const auto* existing = sink->file(); existing && normalized(*existing) == key) {
                result = std::static_pointer_cast<FileSink>(sink);   // the same file twice would be corrupted
                return;
            }
        }
        result = std::make_shared<FileSink>(path, file_options);
        sinks.push_back(result);
        refresh();
    });
    return result;
}

inline ChannelData* Core::channel(std::string_view name) {
    std::lock_guard lock(channels_mutex);
    auto it = channels.find(name);
    if (it == channels.end()) it = channels.emplace(std::string(name), std::make_unique<ChannelData>(std::string(name))).first;
    return it->second.get();
}

inline void Core::shutdown() noexcept {
    try {
        stop_async();
        locked([&] { flush_sinks(); });
    } catch (...) {
    }
}

inline bool enabled(Level level, const ChannelData* channel) noexcept {
    const auto& c = core();
    return level < Level::Off && level >= c.level.load(std::memory_order_relaxed)
           && level >= c.sinks_level.load(std::memory_order_relaxed)
           && (channel == nullptr || level >= channel->level.load(std::memory_order_relaxed));
}

inline void submit(Level level, const ChannelData* channel, const std::source_location& location,
                   std::string_view message, std::string_view ansi_message, std::span<const Field> vars) noexcept {
    auto& c = core();
    if (lock_depth > 0) {   // logged from inside a sink: writing it would dead-lock
        c.dropped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    try {
        Record record;
        record.level = level;
        record.time = std::chrono::system_clock::now();
        record.message = message;
        record.ansi_message = ansi_message;
        record.channel = channel ? std::string_view(channel->name) : std::string_view();
        record.thread = thread_label();
        record.location = location;
        record.context = thread_context().views;
        record.vars = vars;
        if (c.async.load(std::memory_order_acquire)) c.enqueue(record);
        else c.write(record);
    } catch (...) {   // out of memory: the record is lost, the program goes on
        c.dropped.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace detail

// -- Sink members ---------------------------------------------------------------------

inline bool Sink::handle(const Record& record) noexcept {
    if (record.level < level()) return false;
    try {
        if (filter_ && !filter_(record)) return false;
        line_.clear();
        const auto& c = detail::core();
        if (json_) detail::render_json(record, line_);
        else pattern_.render(record, colored(), c.level_styles, c.start, line_);
        line_ += '\n';
        write(record, line_);
        return true;
    } catch (const std::exception& error) {
        detail::internal_error("a sink failed: ", error.what());
    } catch (...) {
        detail::internal_error("a sink failed");
    }
    return false;
}

inline void Sink::set_level(Level minimum) {
    auto& c = detail::core();
    c.locked([&] {
        level_.store(minimum, std::memory_order_relaxed);
        c.refresh();
    });
}

inline void Sink::set_pattern(std::string_view pattern) {
    detail::Pattern compiled(pattern);
    detail::core().locked([&] { pattern_ = std::move(compiled); });
}

inline void Sink::set_json(bool enabled) {
    auto& c = detail::core();
    c.locked([&] {
        json_ = enabled;
        c.refresh();
    });
}

inline void Sink::set_filter(std::function<bool(const Record&)> filter) {
    detail::core().locked([&] { filter_ = std::move(filter); });
}

inline void ConsoleSink::set_color(ColorMode mode) {
    auto& c = detail::core();
    c.locked([&] {
        mode_ = mode;
        color_.store(colors_for(stream(), mode));
        c.refresh();
    });
}

inline void ConsoleSink::set_stream(std::FILE* stream) {
    auto& c = detail::core();
    c.locked([&] {
        flush();
        stream_.store(stream);
        color_.store(colors_for(stream, mode_));
        c.refresh();
    });
}

inline bool FileSink::open(bool truncate) {
    std::error_code error;
    if (path_.has_parent_path()) std::filesystem::create_directories(path_.parent_path(), error);
#if defined(_WIN32)
    file_ = _wfsopen(path_.c_str(), truncate ? L"wb" : L"ab", _SH_DENYNO);   // others may read it meanwhile
#else
    file_ = std::fopen(path_.c_str(), truncate ? "wb" : "ab");
#endif
    if (!file_) {
        fail("cannot open", std::error_code(errno, std::generic_category()));
        retry_ = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        open_.store(false);
        return false;
    }
    std::setvbuf(file_, nullptr, _IOFBF, 64 * 1024);
    size_ = truncate ? 0 : std::filesystem::file_size(path_, error);
    if (error) size_ = 0;
    failing_ = false;
    open_.store(true);
    return true;
}

inline void FileSink::write(const Record&, std::string_view line) {
    if (!file_ && (std::chrono::steady_clock::now() < retry_ || !open(false))) return;
    if (options_.max_size > 0 && size_ > 0 && size_ + line.size() > options_.max_size) {
        rotate();
        if (!file_) return;
    }
    if (std::fwrite(line.data(), 1, line.size(), file_) != line.size()) {
        fail("cannot write to", std::error_code(errno, std::generic_category()));
        std::clearerr(file_);
        return;
    }
    size_ += line.size();
}

inline void FileSink::flush() {
    if (!file_) return;
    if (std::fflush(file_) != 0) {
        fail("cannot write to", std::error_code(errno, std::generic_category()));
        std::clearerr(file_);
    } else {
        failing_ = false;
    }
}

inline void FileSink::rotate() {
    std::fclose(file_);
    file_ = nullptr;
    open_.store(false);
    if (options_.max_files > 0) {
        std::error_code error;
        std::filesystem::remove(rotated(options_.max_files), error);
        for (auto index = options_.max_files; index > 1; --index) {
            const auto from = rotated(index - 1);
            if (std::filesystem::exists(from, error)) std::filesystem::rename(from, rotated(index), error);
        }
        error.clear();
        std::filesystem::rename(path_, rotated(1), error);
        if (error) {   // on Windows, another program may hold the file: keep writing to it
            fail("cannot rotate", error);
            if (open(false)) size_ = 0;   // next attempt after another max_size bytes
            return;
        }
    }
    open(true);
}

inline std::filesystem::path FileSink::rotated(std::size_t index) const {
    auto name = path_.stem();
    name += "." + std::to_string(index);
    name += path_.extension();
    return path_.parent_path() / name;
}

inline void FileSink::fail(std::string_view what, const std::error_code& error) {
    if (failing_) return;
    failing_ = true;
    detail::internal_error(std::format("{} \"{}\": ", what, detail::path_text(path_)),
                           error.default_error_condition().message());
}

// -- Front end ------------------------------------------------------------------------

namespace detail {

struct Scratch {
    std::string message, ansi_message, format;
    bool busy = false;
};

// Per-thread buffers reused from one log to the next: no allocation once warm. A log made while another one
// is being formatted (from an operator<<) gets buffers of its own.
class ScratchLease {
public:
    ScratchLease() : scratch_(shared().busy ? own_ : shared()) {
        scratch_.busy = true;
        scratch_.message.clear();
        scratch_.ansi_message.clear();
        scratch_.format.clear();
    }
    ~ScratchLease() { scratch_.busy = false; }
    ScratchLease(const ScratchLease&) = delete;
    ScratchLease& operator=(const ScratchLease&) = delete;

    Scratch* operator->() noexcept { return &scratch_; }

private:
    static Scratch& shared() {
        thread_local Scratch scratch;
        return scratch;
    }

    Scratch own_;
    Scratch& scratch_;
};

template <class... Args>
void log(const ChannelData* channel, Level level, const Format<Args...>& format, const Args&... args) {
    if (!enabled(level, channel)) return;
    ScratchLease scratch;
    try {
        if (format.tags) {
            render_tags(format.text, true, false, scratch->format);
            format_into(scratch->message, scratch->format, args...);
            if (core().any_color.load(std::memory_order_relaxed)) {
                scratch->format.clear();
                render_tags(format.text, true, true, scratch->format);
                format_into(scratch->ansi_message, scratch->format, args...);
            }
        } else {
            format_into(scratch->message, format.text, args...);
        }
    } catch (const std::exception& error) {   // a dynamic width out of range, a throwing operator<<...
        scratch->message = std::format("{} [format error: {}]", format.text, error.what());
        scratch->ansi_message.clear();
    } catch (...) {
        scratch->message = std::format("{} [format error]", format.text);
        scratch->ansi_message.clear();
    }
    const std::string_view message = scratch->message;
    submit(level, channel, format.location, message,
           scratch->ansi_message.empty() ? message : std::string_view(scratch->ansi_message), {});
}

// A message without format string: written as it is (a std::string read from a file keeps its braces).
template <class T>
void log_value(const ChannelData* channel, Level level, const std::source_location& location, const T& value) {
    if (!enabled(level, channel)) return;
    if constexpr (std::is_convertible_v<const T&, std::string_view> && !CharPointer<T>) {
        const std::string_view text(value);
        submit(level, channel, location, text, text, {});
    } else {
        ScratchLease scratch;
        try {
            format_into(scratch->message, "{}", value);
        } catch (...) {
            scratch->message = "[format error]";
        }
        submit(level, channel, location, scratch->message, scratch->message, {});
    }
}

template <class... Args>
void print(bool newline, const Format<Args...>& format, const Args&... args) {
    auto& c = core();
    std::string text;
    try {
        if (format.tags) {
            std::string tagless;
            render_tags(format.text, true, c.print_colors(), tagless);
            format_into(text, tagless, args...);
        } else {
            format_into(text, format.text, args...);
        }
    } catch (const std::exception& error) {
        text = std::format("{} [format error: {}]", format.text, error.what());
    } catch (...) {
        text = std::format("{} [format error]", format.text);
    }
    if (newline) text += '\n';
    c.print(text);
}

} // namespace detail

// -- Configuration --------------------------------------------------------------------

inline void set_level(Level minimum) noexcept { detail::core().level.store(minimum, std::memory_order_relaxed); }
[[nodiscard]] inline Level level() noexcept { return detail::core().level.load(std::memory_order_relaxed); }

// True when a record of this level would be written: guards an expensive computation.
[[nodiscard]] inline bool enabled(Level severity) noexcept { return detail::enabled(severity, nullptr); }

// Pattern of every sink that does not write JSON. Throws std::invalid_argument for a wrong pattern.
inline void set_pattern(std::string_view pattern) { detail::core().set_pattern(detail::Pattern(pattern)); }

// Style of a level for the <level> tag: names separated by spaces, e.g. "bold magenta" or "white on_blue".
inline void set_level_style(Level severity, std::string_view style) {
    std::string codes;
    while (!style.empty()) {
        const auto end = (std::min)(style.find(' '), style.size());
        const auto word = style.substr(0, end);
        style.remove_prefix((std::min)(end + 1, style.size()));
        if (word.empty()) continue;
        const auto* found = detail::find_style(word);
        if (!found) throw std::invalid_argument(std::format("Gem::Log: unknown style \"{}\"", word));
        codes += found->code;
    }
    if (severity >= Level::Off) return;
    auto& c = detail::core();
    c.locked([&] { c.level_styles[static_cast<std::size_t>(severity)] = std::move(codes); });
}

// The console sink created at start-up, on stdout.
[[nodiscard]] inline std::shared_ptr<ConsoleSink> console() {
    auto& c = detail::core();
    std::shared_ptr<ConsoleSink> sink;
    c.locked([&] { sink = c.console; });
    return sink;
}

// Adds a log file. Adding the same file again returns its sink.
inline std::shared_ptr<FileSink> add_file(const std::filesystem::path& path, FileOptions options = {}) {
    return detail::core().add_file(path, options);
}

inline void add_sink(std::shared_ptr<Sink> sink) {
    if (!sink) return;
    auto& c = detail::core();
    c.locked([&] {
        if (std::ranges::find(c.sinks, sink) == c.sinks.end()) c.sinks.push_back(std::move(sink));
        c.refresh();
    });
}

inline std::shared_ptr<CallbackSink> add_callback(CallbackSink::Callback callback) {
    auto sink = std::make_shared<CallbackSink>(std::move(callback));
    add_sink(sink);
    return sink;
}

// Removes a sink, the console included: Gem::Log::remove_sink(Gem::Log::console()).
inline void remove_sink(const std::shared_ptr<Sink>& sink) {
    auto& c = detail::core();
    c.drain();   // its queued records first
    c.locked([&] {
        std::erase(c.sinks, sink);
        c.refresh();
    });
}

// Async mode: a background thread writes the records, so logging no longer waits for the console or the disk.
// Records at or above options.flush_level still wait until written. set_async(false) writes what is queued.
inline void set_async(bool on, AsyncOptions options = {}) {
    if (on) detail::core().start_async(options);
    else detail::core().stop_async();
}

// Waits until every record logged so far is written and flushed.
inline void flush() { detail::core().flush(); }

[[nodiscard]] inline Stats stats() {
    auto& c = detail::core();
    Stats result{c.logged.load(std::memory_order_relaxed), c.dropped.load(std::memory_order_relaxed), 0};
    std::lock_guard lock(c.queue_mutex);
    result.queued = c.queue.size();
    return result;
}

// Name shown by %(thread) for the calling thread instead of its id ("" restores the id).
inline void set_thread_name(std::string_view name) {
    detail::thread_label() = name.empty() ? detail::thread_id_label() : std::string(name);
}

// -- Logging --------------------------------------------------------------------------

// Each level gets two overloads:
//   info("format {}", args...)   string literal: a std::format string checked at compile time
//   info(text)                   any other value: written as it is
#define GEMLOG_DEFINE_LEVEL_(name, severity, channel_data, qualifier)                                          \
    template <class... Args>                                                                                   \
    void name(detail::Format<Args...> format, const Args&... args) qualifier {                                 \
        detail::log(channel_data, severity, format, args...);                                                  \
    }                                                                                                          \
    template <class T>                                                                                         \
        requires(!std::is_array_v<T> && !std::is_same_v<T, RuntimeFormat>)                                    \
    void name(const T& message, std::source_location location = std::source_location::current()) qualifier {   \
        detail::log_value(channel_data, severity, location, message);                                          \
    }

GEMLOG_DEFINE_LEVEL_(trace, Level::Trace, nullptr, )
GEMLOG_DEFINE_LEVEL_(debug, Level::Debug, nullptr, )
GEMLOG_DEFINE_LEVEL_(info, Level::Info, nullptr, )
GEMLOG_DEFINE_LEVEL_(success, Level::Success, nullptr, )
GEMLOG_DEFINE_LEVEL_(warning, Level::Warning, nullptr, )
GEMLOG_DEFINE_LEVEL_(error, Level::Error, nullptr, )
GEMLOG_DEFINE_LEVEL_(critical, Level::Critical, nullptr, )

// Level chosen at run time: Gem::Log::log(level, "format {}", args...).
template <class... Args>
void log(Level severity, detail::Format<Args...> format, const Args&... args) {
    detail::log(nullptr, severity, format, args...);
}

template <class T>
    requires(!std::is_array_v<T> && !std::is_same_v<T, RuntimeFormat>)
void log(Level severity, const T& message, std::source_location location = std::source_location::current()) {
    detail::log_value(nullptr, severity, location, message);
}

// A named logger for a subsystem. Its name appears in %(channel), its level filters only its own records.
// Channels with the same name share their settings.
//     Gem::Log::Channel net{"net"};
//     net.set_level(Gem::Log::Level::Warning);
//     net.error("Connection to {} lost", host);
class Channel {
public:
    explicit Channel(std::string_view name) : data_(detail::core().channel(name)) {}

    [[nodiscard]] std::string_view name() const noexcept { return data_->name; }
    void set_level(Level minimum) noexcept { data_->level.store(minimum, std::memory_order_relaxed); }
    [[nodiscard]] Level level() const noexcept { return data_->level.load(std::memory_order_relaxed); }
    [[nodiscard]] bool enabled(Level severity) const noexcept { return detail::enabled(severity, data_); }

    template <class... Args>
    void log(Level severity, detail::Format<Args...> format, const Args&... args) const {
        detail::log(data_, severity, format, args...);
    }

    GEMLOG_DEFINE_LEVEL_(trace, Level::Trace, data_, const)
    GEMLOG_DEFINE_LEVEL_(debug, Level::Debug, data_, const)
    GEMLOG_DEFINE_LEVEL_(info, Level::Info, data_, const)
    GEMLOG_DEFINE_LEVEL_(success, Level::Success, data_, const)
    GEMLOG_DEFINE_LEVEL_(warning, Level::Warning, data_, const)
    GEMLOG_DEFINE_LEVEL_(error, Level::Error, data_, const)
    GEMLOG_DEFINE_LEVEL_(critical, Level::Critical, data_, const)

private:
    detail::ChannelData* data_;
};

#undef GEMLOG_DEFINE_LEVEL_

// Adds key=value to every record of this thread while it lives: %(context) in patterns, "context" in JSON.
//     Gem::Log::Context request{"request", id};
class Context {
public:
    template <class T>
    Context(std::string_view key, const T& value) {
        detail::OwnedField field{std::string(key), {}, FieldKind::String};
        try {
            detail::field_value(value, field.value, field.kind);
        } catch (...) {   // a throwing operator<<
            field.value = "[format error]";
            field.kind = FieldKind::Other;
        }
        auto& context = detail::thread_context();
        id_ = ++context.next_id;
        context.entries.push_back({id_, std::move(field)});
        context.rebuild();
    }
    ~Context() {
        auto& context = detail::thread_context();
        std::erase_if(context.entries, [&](const auto& entry) { return entry.id == id_; });
        context.rebuild();
    }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

private:
    std::uint64_t id_ = 0;
};

// Logs the time spent from its creation to the end of its scope: "load textures: 12.48 ms".
class Timer {
public:
    explicit Timer(std::string_view label, Level severity = Level::Debug,
                   std::source_location location = std::source_location::current())
        : label_(label), level_(severity), location_(location), start_(std::chrono::steady_clock::now()) {}

    ~Timer() {
        if (!detail::enabled(level_, nullptr)) return;
        try {
            const auto ns = static_cast<double>(elapsed().count());
            std::string message;
            if (ns < 1e6) message = std::format("{}: {:.1f} us", label_, ns / 1e3);
            else if (ns < 1e9) message = std::format("{}: {:.2f} ms", label_, ns / 1e6);
            else message = std::format("{}: {:.3f} s", label_, ns / 1e9);
            detail::submit(level_, nullptr, location_, message, message, {});
        } catch (...) {
        }
    }

    [[nodiscard]] std::chrono::nanoseconds elapsed() const noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start_);
    }

    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

private:
    std::string label_;
    Level level_;
    std::source_location location_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace Gem::Log

namespace Gem {

using LogLevel = Log::Level;

// std::print with color tags: Gem::println("<green>OK</green> {} files", n). Thread-safe, never mixed with a
// log line, kept in order with the logs in async mode. Tags are removed when stdout is not a terminal.
template <class... Args>
void print(Log::detail::Format<Args...> format, const Args&... args) {
    Log::detail::print(false, format, args...);
}

template <class... Args>
void println(Log::detail::Format<Args...> format, const Args&... args) {
    Log::detail::print(true, format, args...);
}

inline void println() { Log::detail::core().print("\n"); }

} // namespace Gem

// -- Macros ---------------------------------------------------------------------------

#define GEMLOG_LEVEL_TRACE 0
#define GEMLOG_LEVEL_DEBUG 1
#define GEMLOG_LEVEL_INFO 2
#define GEMLOG_LEVEL_SUCCESS 3
#define GEMLOG_LEVEL_WARNING 4
#define GEMLOG_LEVEL_ERROR 5
#define GEMLOG_LEVEL_CRITICAL 6
#define GEMLOG_LEVEL_OFF 7

#ifndef GEMLOG_LEVEL
#   define GEMLOG_LEVEL GEMLOG_LEVEL_TRACE
#endif

#ifndef GEMLOG_NO_MACROS

#define GEMLOG_CONCAT_IMPL_(a, b) a##b
#define GEMLOG_CONCAT_(a, b) GEMLOG_CONCAT_IMPL_(a, b)
// The arguments are evaluated only when the level is on.
#define GEMLOG_IF_(severity, ...) (::Gem::Log::enabled(severity) ? __VA_ARGS__ : void())
#define GEMLOG_ONCE_(...)                                                                                      \
    do {                                                                                                       \
        static ::std::atomic_flag gemlog_once_;                                                                \
        if (!gemlog_once_.test_and_set(::std::memory_order_relaxed)) { __VA_ARGS__; }                          \
    } while (false)

#if GEMLOG_LEVEL <= GEMLOG_LEVEL_TRACE
#   define LOG_TRACE(...) GEMLOG_IF_(::Gem::Log::Level::Trace, ::Gem::Log::trace(__VA_ARGS__))
#   define LOG_TRACE_ONCE(...) GEMLOG_ONCE_(LOG_TRACE(__VA_ARGS__))
#else
#   define LOG_TRACE(...) ((void)0)
#   define LOG_TRACE_ONCE(...) ((void)0)
#endif

#if GEMLOG_LEVEL <= GEMLOG_LEVEL_DEBUG
#   define LOG_DEBUG(...) GEMLOG_IF_(::Gem::Log::Level::Debug, ::Gem::Log::debug(__VA_ARGS__))
#   define LOG_DEBUG_ONCE(...) GEMLOG_ONCE_(LOG_DEBUG(__VA_ARGS__))
#   define LOG_DUMP(...)                                                                                       \
        GEMLOG_IF_(::Gem::Log::Level::Debug,                                                                   \
                   ::Gem::Log::detail::dump(::std::source_location::current(), #__VA_ARGS__, __VA_ARGS__))
#   define LOG_TIMER(...) ::Gem::Log::Timer GEMLOG_CONCAT_(gemlog_timer_, __COUNTER__){__VA_ARGS__}
#else
#   define LOG_DEBUG(...) ((void)0)
#   define LOG_DEBUG_ONCE(...) ((void)0)
#   define LOG_DUMP(...) ((void)0)
#   define LOG_TIMER(...) ((void)0)
#endif

#if GEMLOG_LEVEL <= GEMLOG_LEVEL_INFO
#   define LOG_INFO(...) GEMLOG_IF_(::Gem::Log::Level::Info, ::Gem::Log::info(__VA_ARGS__))
#   define LOG_INFO_ONCE(...) GEMLOG_ONCE_(LOG_INFO(__VA_ARGS__))
#else
#   define LOG_INFO(...) ((void)0)
#   define LOG_INFO_ONCE(...) ((void)0)
#endif

#if GEMLOG_LEVEL <= GEMLOG_LEVEL_SUCCESS
#   define LOG_SUCCESS(...) GEMLOG_IF_(::Gem::Log::Level::Success, ::Gem::Log::success(__VA_ARGS__))
#   define LOG_SUCCESS_ONCE(...) GEMLOG_ONCE_(LOG_SUCCESS(__VA_ARGS__))
#else
#   define LOG_SUCCESS(...) ((void)0)
#   define LOG_SUCCESS_ONCE(...) ((void)0)
#endif

#if GEMLOG_LEVEL <= GEMLOG_LEVEL_WARNING
#   define LOG_WARNING(...) GEMLOG_IF_(::Gem::Log::Level::Warning, ::Gem::Log::warning(__VA_ARGS__))
#   define LOG_WARNING_ONCE(...) GEMLOG_ONCE_(LOG_WARNING(__VA_ARGS__))
#else
#   define LOG_WARNING(...) ((void)0)
#   define LOG_WARNING_ONCE(...) ((void)0)
#endif

#if GEMLOG_LEVEL <= GEMLOG_LEVEL_ERROR
#   define LOG_ERROR(...) GEMLOG_IF_(::Gem::Log::Level::Error, ::Gem::Log::error(__VA_ARGS__))
#   define LOG_ERROR_ONCE(...) GEMLOG_ONCE_(LOG_ERROR(__VA_ARGS__))
#else
#   define LOG_ERROR(...) ((void)0)
#   define LOG_ERROR_ONCE(...) ((void)0)
#endif

#if GEMLOG_LEVEL <= GEMLOG_LEVEL_CRITICAL
#   define LOG_CRITICAL(...) GEMLOG_IF_(::Gem::Log::Level::Critical, ::Gem::Log::critical(__VA_ARGS__))
#   define LOG_CRITICAL_ONCE(...) GEMLOG_ONCE_(LOG_CRITICAL(__VA_ARGS__))
#else
#   define LOG_CRITICAL(...) ((void)0)
#   define LOG_CRITICAL_ONCE(...) ((void)0)
#endif

#define LOG_CONTEXT(key, ...) ::Gem::Log::Context GEMLOG_CONCAT_(gemlog_context_, __COUNTER__){key, __VA_ARGS__}

#endif // GEMLOG_NO_MACROS
