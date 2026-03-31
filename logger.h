#pragma once
// Gem::Logger – Thread-safe logging system (C++23)
// 7 levels, @{key} interpolation, JSON output, color DSL, file rotation, lock-free queue

#ifdef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#else
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <deque>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <source_location>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#ifdef __cpp_lib_hardware_interference_size
inline constexpr std::size_t hardware_destructive_interference_size =
std::hardware_destructive_interference_size;
#else
#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(__powerpc__)
inline constexpr std::size_t hardware_destructive_interference_size = 128;
#else
inline constexpr std::size_t hardware_destructive_interference_size = 64;
#endif
#endif

namespace Gem
{

    // ── Core types ───────────────────────────────────────────────────

    using ContextValue = std::variant<bool, int, long long, double, std::string>;
    using ContextMap = std::unordered_map<std::string, ContextValue>;

    enum class LogLevel : std::uint8_t {
        Trace = 0, Debug, Info, Success, Warning, Error, Critical
    };

    [[nodiscard]] constexpr std::string_view level_to_string(LogLevel level) noexcept {
        constexpr std::string_view names[] = {
            "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
        };
        auto i = static_cast<std::uint8_t>(level);
        return i <= 6 ? names[i] : "UNKNOWN";
    }

    // ── Error types ──────────────────────────────────────────────────

    enum class FileError : std::uint8_t {
        DirectoryCreationFailed = 1, FileOpenFailed, WriteFailed,
        FlushFailed, RotationFailed, PermissionDenied, DiskFull, CacheFull
    };

    [[nodiscard]] constexpr std::string_view file_error_to_string(FileError e) noexcept {
        switch (e) {
        case FileError::DirectoryCreationFailed: return "Failed to create directory";
        case FileError::FileOpenFailed:          return "Failed to open file";
        case FileError::WriteFailed:             return "Write operation failed";
        case FileError::FlushFailed:             return "Flush operation failed";
        case FileError::RotationFailed:          return "File rotation failed";
        case FileError::PermissionDenied:        return "Permission denied";
        case FileError::DiskFull:                return "Disk full";
        case FileError::CacheFull:               return "File cache full";
        default:                                 return "Unknown error";
        }
    }

    enum class ConfigError : std::uint8_t {
        InvalidName = 1, InvalidFormat, MissingFormat, NoOutputs, TooManyHandlers
    };

    // ── Helpers ──────────────────────────────────────────────────────

    inline std::string value_to_string(const ContextValue& v) {
        return std::visit([](const auto& val) -> std::string {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, bool>)             return val ? "true" : "false";
            else if constexpr (std::is_same_v<T, std::string>) return val;
            else                                               return std::to_string(val);
            }, v);
    }

    // @{key} interpolation — @@ escapes to literal @
    inline std::string interpolate_message(std::string_view tpl, const ContextMap& ctx) {
        if (tpl.find('@') == std::string_view::npos)
            return std::string(tpl);

        std::string result;
        result.reserve(tpl.size() * 2);
        std::size_t pos = 0;

        while (pos < tpl.size()) {
            auto at = tpl.find('@', pos);
            if (at == std::string_view::npos) { result.append(tpl.substr(pos)); break; }

            result.append(tpl.substr(pos, at - pos));

            if (at + 1 < tpl.size() && tpl[at + 1] == '@') {
                result += '@'; pos = at + 2; continue;
            }
            if (at + 1 < tpl.size() && tpl[at + 1] == '{') {
                auto close = tpl.find('}', at + 2);
                if (close != std::string_view::npos) {
                    auto key = tpl.substr(at + 2, close - at - 2);
                    if (auto it = ctx.find(std::string(key)); it != ctx.end())
                        result.append(value_to_string(it->second));
                    else
                        result.append(tpl.substr(at, close - at + 1));
                    pos = close + 1; continue;
                }
            }
            result += '@'; pos = at + 1;
        }
        return result;
    }

    inline std::string json_escape(std::string_view str) {
        std::string result;
        result.reserve(str.size() + str.size() / 8);
        for (unsigned char ch : str) {
            switch (ch) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b";  break;
            case '\f': result += "\\f";  break;
            case '\n': result += "\\n";  break;
            case '\r': result += "\\r";  break;
            case '\t': result += "\\t";  break;
            default:
                if (ch < 0x20) result += std::format("\\u{:04x}", ch);
                else           result += ch; // valid UTF-8 passes through as valid JSON
            }
        }
        return result;
    }

    // ── StreamTarget ─────────────────────────────────────────────────

    class StreamTarget {
    public:
        StreamTarget() noexcept = default;
        explicit StreamTarget(std::ostream* s) noexcept : stream_(s) {}
        explicit StreamTarget(std::shared_ptr<std::ostream> s) noexcept : stream_(std::move(s)) {}

        [[nodiscard]] std::ostream* get() const noexcept {
            return std::visit([](auto&& arg) -> std::ostream* {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::monostate>) return nullptr;
                else if constexpr (std::is_same_v<T, std::ostream*>) return arg;
                else return arg.get();
                }, stream_);
        }

        [[nodiscard]] static StreamTarget cout() noexcept { return StreamTarget(&std::cout); }
        [[nodiscard]] static StreamTarget cerr() noexcept { return StreamTarget(&std::cerr); }

    private:
        std::variant<std::monostate, std::ostream*, std::shared_ptr<std::ostream>> stream_;
    };

    // ── FileCache ────────────────────────────────────────────────────

    struct FileCacheConfig {
        std::chrono::milliseconds idle_timeout{ 30000 };
        std::size_t max_file_size{ 100 * 1024 * 1024 };
        std::size_t max_cached_files{ 64 };
        bool auto_flush{ true };
        bool enable_rotation{ true };
    };

    class FileCache {
    public:
        using Config = FileCacheConfig;

        explicit FileCache(Config config = {});

        ~FileCache() {
            if (cleanup_thread_ && cleanup_thread_->joinable()) {
                cleanup_thread_->request_stop();
                cleanup_thread_->join();
            }
            std::unique_lock lock(mutex_);
            cache_.clear();
        }

        [[nodiscard]] std::expected<void, FileError> write(const std::filesystem::path& path,
            std::string_view message) {
            std::unique_lock lock(mutex_);

            auto it = cache_.find(path.string());
            if (it == cache_.end()) {
                if (cache_.size() >= config_.max_cached_files)
                    return std::unexpected(FileError::CacheFull);

                if (auto r = create_dirs(path); !r) return r;

                auto stream = std::make_unique<std::ofstream>(path, std::ios::app);
                if (!stream->is_open())
                    return std::unexpected(FileError::FileOpenFailed);

                auto entry = std::make_unique<Entry>();
                entry->stream = std::move(stream);
                entry->last_access = std::chrono::steady_clock::now();
                it = cache_.emplace(path.string(), std::move(entry)).first;
            }

            auto& entry = *it->second;
            entry.last_access = std::chrono::steady_clock::now();

            if (config_.enable_rotation &&
                entry.current_size + message.size() > config_.max_file_size) {
                if (auto r = rotate(path, entry); !r) return r;
            }

            *entry.stream << message;
            if (!entry.stream->good())
                return std::unexpected(FileError::WriteFailed);

            entry.current_size += message.size();
            if (config_.auto_flush) entry.stream->flush();

            return {};
        }

        [[nodiscard]] std::expected<void, FileError> flush_all() {
            std::unique_lock lock(mutex_);
            for (auto& [_, e] : cache_) {
                e->stream->flush();
                if (!e->stream->good())
                    return std::unexpected(FileError::FlushFailed);
            }
            return {};
        }

    private:
        struct Entry {
            std::unique_ptr<std::ofstream> stream;
            std::chrono::steady_clock::time_point last_access;
            std::size_t current_size = 0;
        };

        std::expected<void, FileError> create_dirs(const std::filesystem::path& path) {
            try {
                auto parent = path.parent_path();
                if (!parent.empty() && !std::filesystem::exists(parent)) {
                    std::error_code ec;
                    if (!std::filesystem::create_directories(parent, ec))
                        return std::unexpected(FileError::DirectoryCreationFailed);
                }
                return {};
            }
            catch (...) {
                return std::unexpected(FileError::DirectoryCreationFailed);
            }
        }

        std::expected<void, FileError> rotate(const std::filesystem::path& path, Entry& entry) {
            entry.stream->close();

            auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::tm tm{};
#ifdef _WIN32
            localtime_s(&tm, &tt);
#else
            localtime_r(&tt, &tm);
#endif
            std::ostringstream ss;
            ss << path.stem().string() << "_" << std::put_time(&tm, "%Y%m%d_%H%M%S")
                << path.extension().string();

            std::error_code ec;
            std::filesystem::rename(path, path.parent_path() / ss.str(), ec);
            if (ec) return std::unexpected(FileError::RotationFailed);

            entry.stream = std::make_unique<std::ofstream>(path, std::ios::app);
            if (!entry.stream->is_open())
                return std::unexpected(FileError::FileOpenFailed);

            entry.current_size = 0;
            return {};
        }

        void cleanup_expired() {
            std::unique_lock lock(mutex_);
            auto now = std::chrono::steady_clock::now();
            std::erase_if(cache_, [&](const auto& p) {
                return now - p.second->last_access > config_.idle_timeout;
                });
        }

        mutable std::shared_mutex mutex_;
        std::unordered_map<std::string, std::unique_ptr<Entry>> cache_;
        Config config_;
        std::mutex sleep_mutex_;
        std::condition_variable_any sleep_cv_;
        std::optional<std::jthread> cleanup_thread_;
    };

    inline FileCache::FileCache(Config config) : config_(config) {
        cleanup_thread_.emplace([this](std::stop_token st) {
            while (!st.stop_requested()) {
                std::unique_lock lock(sleep_mutex_);
                sleep_cv_.wait_for(lock, st, std::chrono::seconds(10), [] { return false; });
                if (!st.stop_requested()) cleanup_expired();
            }
            });
    }

    // Intentional leak: avoids static destruction order fiasco.
    // LoggerImpl may flush to FileCache during its own static destructor.
    // OS reclaims file handles at process exit.
    inline FileCache& get_file_cache() {
        static auto* instance = new FileCache;
        return *instance;
    }

    // ── LogRecord ────────────────────────────────────────────────────

    struct LogRecord {
        LogLevel level;
        std::string message;
        std::string handler_hint;
        std::chrono::system_clock::time_point timestamp;
        ContextMap context;
        std::optional<std::string> lexeme;
        std::source_location location;
        std::thread::id thread_id;

        LogRecord(LogLevel lvl, std::string msg, std::string_view hint = {},
            std::source_location loc = std::source_location::current())
            : level(lvl), message(std::move(msg)), handler_hint(hint),
            timestamp(std::chrono::system_clock::now()), location(loc),
            thread_id(std::this_thread::get_id()) {
        }

        LogRecord(LogRecord&&) noexcept = default;
        LogRecord& operator=(LogRecord&&) noexcept = default;
        LogRecord(const LogRecord&) = delete;
        LogRecord& operator=(const LogRecord&) = delete;

        // Default: only for LockFreeQueue cell initialization
        LogRecord() : level(LogLevel::Info), timestamp{}, location{}, thread_id{} {}
    };

    // ── Format engine ────────────────────────────────────────────────

    enum class FormatToken : std::uint8_t {
        Literal = 0, Level, Date, Time, Message, Thread, File, Line, Function, Context
    };

    struct FormatSegment { FormatToken token; std::string content; };

    class ParsedFormat {
    public:
        static ParsedFormat parse(std::string_view pattern) {
            ParsedFormat fmt;
            std::size_t pos = 0;

            while (pos < pattern.size()) {
                auto ts = pattern.find("%(", pos);
                if (ts == std::string_view::npos) {
                    if (pos < pattern.size())
                        fmt.segments_.emplace_back(FormatToken::Literal, std::string(pattern.substr(pos)));
                    break;
                }
                if (ts > pos)
                    fmt.segments_.emplace_back(FormatToken::Literal, std::string(pattern.substr(pos, ts - pos)));

                auto te = pattern.find(')', ts);
                if (te == std::string_view::npos) { fmt.valid_ = false; return fmt; }

                auto tok = pattern.substr(ts + 2, te - ts - 2);
                if (tok.starts_with("context[") && tok.ends_with("]")) {
                    fmt.segments_.emplace_back(FormatToken::Context, std::string(tok.substr(8, tok.size() - 9)));
                }
                else if (auto t = token_from(tok)) {
                    fmt.segments_.emplace_back(*t, "");
                }
                else { fmt.valid_ = false; return fmt; }

                pos = te + 1;
            }
            return fmt;
        }

        [[nodiscard]] std::string apply(const LogRecord& record, const ContextMap& base = {}) const {
            if (!valid_) return "[FORMAT ERROR]";

            auto tt = std::chrono::system_clock::to_time_t(record.timestamp);
            std::tm tm{};
#ifdef _WIN32
            localtime_s(&tm, &tt);
#else
            localtime_r(&tt, &tm);
#endif
            std::ostringstream out;
            out.imbue(std::locale::classic());

            for (const auto& s : segments_) {
                switch (s.token) {
                case FormatToken::Literal:  out << s.content; break;
                case FormatToken::Level:    out << level_to_string(record.level); break;
                case FormatToken::Date:     out << std::put_time(&tm, "%Y-%m-%d"); break;
                case FormatToken::Time:     out << std::put_time(&tm, "%H:%M:%S"); break;
                case FormatToken::Message:  out << record.message; break;
                case FormatToken::Thread:   out << record.thread_id; break;
                case FormatToken::File:     out << record.location.file_name(); break;
                case FormatToken::Line:     out << record.location.line(); break;
                case FormatToken::Function: out << record.location.function_name(); break;
                case FormatToken::Context: {
                    if (auto it = record.context.find(s.content); it != record.context.end())
                        out << value_to_string(it->second);
                    else if (auto it2 = base.find(s.content); it2 != base.end())
                        out << value_to_string(it2->second);
                    break;
                }
                }
            }
            return out.str();
        }

    private:
        std::vector<FormatSegment> segments_;
        bool valid_ = true;

        static std::optional<FormatToken> token_from(std::string_view s) {
            static const std::unordered_map<std::string_view, FormatToken> map = {
                {"levelname", FormatToken::Level},    {"date", FormatToken::Date},
                {"time", FormatToken::Time},          {"message", FormatToken::Message},
                {"thread", FormatToken::Thread},      {"file", FormatToken::File},
                {"line", FormatToken::Line},           {"function", FormatToken::Function}
            };
            auto it = map.find(s);
            return it != map.end() ? std::optional(it->second) : std::nullopt;
        }
    };

    // ── Color DSL ────────────────────────────────────────────────────

    class ColorProcessor {
    public:
        [[nodiscard]] static std::string process(std::string_view input) {
            std::string result;
            result.reserve(input.size() + input.size() / 2);
            std::size_t pos = 0;

            while (pos < input.size()) {
                auto ts = input.find('<', pos);
                if (ts == std::string_view::npos) { result.append(input.substr(pos)); break; }

                result.append(input.substr(pos, ts - pos));
                auto te = input.find('>', ts);
                if (te == std::string_view::npos) { result.append(input.substr(ts)); break; }

                auto tag = input.substr(ts + 1, te - ts - 1);
                bool closing = tag.starts_with('/');
                if (closing) tag = tag.substr(1);

                bool found = false;
                for (const auto& [name, open, close] : colors) {
                    if (tag == name) { result.append(closing ? close : open); found = true; break; }
                }
                if (!found) result.append(input.substr(ts, te - ts + 1));
                pos = te + 1;
            }
            return result;
        }

    private:
        struct Def { std::string_view name, open, close; };
        static constexpr std::array<Def, 11> colors = { {
            {"red","\033[31m","\033[0m"},      {"green","\033[32m","\033[0m"},
            {"yellow","\033[33m","\033[0m"},   {"blue","\033[34m","\033[0m"},
            {"magenta","\033[35m","\033[0m"},  {"cyan","\033[36m","\033[0m"},
            {"white","\033[37m","\033[0m"},    {"bold","\033[1m","\033[22m"},
            {"dim","\033[2m","\033[22m"},      {"italic","\033[3m","\033[23m"},
            {"underline","\033[4m","\033[24m"}
        } };
    };

    // ── Config builder ───────────────────────────────────────────────

    struct LogRecord;
    class ConfigTemplate;

    namespace states { struct Init {}; struct Named {}; struct Leveled {}; struct Formatted {}; struct Complete {}; }

    struct ConfigData {
        using Output = std::variant<StreamTarget, std::filesystem::path>;
        using Filter = std::function<bool(const LogRecord&)>;

        std::string name;
        LogLevel min_level = LogLevel::Trace;
        std::vector<Filter> filters;
        std::unordered_map<std::string, ParsedFormat> formats;
        std::vector<std::pair<std::string, Output>> outputs;
        ContextMap base_context;
        bool structured = false;
    };

    template<typename S = states::Init>
    class ConfigBuilder {
    public:
        ConfigBuilder() = default;
        explicit ConfigBuilder(std::shared_ptr<ConfigData> d) : d_(std::move(d)) {}

        template<typename X = S> requires std::is_same_v<X, states::Init>
        [[nodiscard]] auto name(std::string_view n) const {
            auto d = std::make_shared<ConfigData>();
            d->name = n;
            return ConfigBuilder<states::Named>(d);
        }

        template<typename X = S> requires std::is_same_v<X, states::Named>
        [[nodiscard]] auto level(LogLevel l) const {
            d_->min_level = l;
            return ConfigBuilder<states::Leveled>(d_);
        }

        template<typename X = S> requires (std::is_same_v<X, states::Leveled> || std::is_same_v<X, states::Formatted>)
            [[nodiscard]] auto format(std::string_view n, std::string_view p) const {
            d_->formats[std::string(n)] = ParsedFormat::parse(p);
            return ConfigBuilder<states::Formatted>(d_);
        }

        template<typename X = S> requires (std::is_same_v<X, states::Formatted> || std::is_same_v<X, states::Complete>)
            [[nodiscard]] auto output(std::string_view fn, ConfigData::Output t) const {
            d_->outputs.emplace_back(std::string(fn), std::move(t));
            return ConfigBuilder<states::Complete>(d_);
        }

        template<typename X = S> requires (!std::is_same_v<X, states::Init> && !std::is_same_v<X, states::Named>)
            [[nodiscard]] auto filter(ConfigData::Filter f) const { d_->filters.push_back(std::move(f)); return ConfigBuilder<X>(d_); }

        template<typename X = S> requires (!std::is_same_v<X, states::Init> && !std::is_same_v<X, states::Named>)
            [[nodiscard]] auto context(ContextMap c) const { d_->base_context = std::move(c); return ConfigBuilder<X>(d_); }

        template<typename X = S> requires (!std::is_same_v<X, states::Init> && !std::is_same_v<X, states::Named>)
            [[nodiscard]] auto structured(bool e = true) const { d_->structured = e; return ConfigBuilder<X>(d_); }

        template<typename X = S> requires std::is_same_v<X, states::Complete>
        [[nodiscard]] ConfigTemplate build() const;

    private:
        std::shared_ptr<ConfigData> d_;
    };

    class ConfigTemplate {
    public:
        using Filter = ConfigData::Filter;
        using Output = ConfigData::Output;

        explicit ConfigTemplate(std::shared_ptr<ConfigData> d) : d_(std::move(d)) {}

        [[nodiscard]] const std::string& name()  const noexcept { return d_->name; }
        [[nodiscard]] LogLevel level()           const noexcept { return d_->min_level; }
        [[nodiscard]] const auto& filters()      const noexcept { return d_->filters; }
        [[nodiscard]] const auto& formats()      const noexcept { return d_->formats; }
        [[nodiscard]] const auto& outputs()      const noexcept { return d_->outputs; }
        [[nodiscard]] const auto& context()      const noexcept { return d_->base_context; }
        [[nodiscard]] bool is_structured()       const noexcept { return d_->structured; }

        [[nodiscard]] static auto builder() { return ConfigBuilder<>(); }

    private:
        std::shared_ptr<ConfigData> d_;
    };

    template<typename S>
    template<typename X> requires std::is_same_v<X, states::Complete>
    ConfigTemplate ConfigBuilder<S>::build() const { return ConfigTemplate(d_); }

    // ── Lock-free queue ──────────────────────────────────────────────

    enum class OverflowPolicy { Block, DropOldest, DropNewest };

    template<typename T>
    class LockFreeQueue {
    public:
        explicit LockFreeQueue(std::size_t cap)
            : cap_(next_pow2(cap)), mask_(cap_ - 1),
            buf_(new Cell[cap_]), head_(0), tail_(0) {
            for (std::size_t i = 0; i < cap_; ++i)
                buf_[i].seq.store(i, std::memory_order_relaxed);
        }
        ~LockFreeQueue() { delete[] buf_; }

        [[nodiscard]] bool try_enqueue(T&& item) {
            std::size_t pos = head_.load(std::memory_order_relaxed);
            for (;;) {
                auto& cell = buf_[pos & mask_];
                auto seq = cell.seq.load(std::memory_order_acquire);
                auto diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
                if (diff == 0) {
                    if (head_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                        cell.data = std::move(item);
                        cell.seq.store(pos + 1, std::memory_order_release);
                        return true;
                    }
                }
                else if (diff < 0) return false;
                else pos = head_.load(std::memory_order_relaxed);
            }
        }

        [[nodiscard]] bool try_dequeue(T& item) {
            std::size_t pos = tail_.load(std::memory_order_relaxed);
            for (;;) {
                auto& cell = buf_[pos & mask_];
                auto seq = cell.seq.load(std::memory_order_acquire);
                auto diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
                if (diff == 0) {
                    if (tail_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                        item = std::move(cell.data);
                        cell.seq.store(pos + mask_ + 1, std::memory_order_release);
                        return true;
                    }
                }
                else if (diff < 0) return false;
                else pos = tail_.load(std::memory_order_relaxed);
            }
        }

        [[nodiscard]] std::size_t size() const noexcept {
            auto h = head_.load(std::memory_order_relaxed);
            auto t = tail_.load(std::memory_order_relaxed);
            return h > t ? h - t : 0;
        }

        LockFreeQueue(const LockFreeQueue&) = delete;
        LockFreeQueue& operator=(const LockFreeQueue&) = delete;

    private:
        struct Cell {
            alignas(hardware_destructive_interference_size) std::atomic<std::size_t> seq;
            T data;
            Cell() : seq(0), data{} {}
        };

        static std::size_t next_pow2(std::size_t n) {
            n--; n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16; n |= n >> 32; return ++n;
        }

        const std::size_t cap_, mask_;
        Cell* const buf_;
        alignas(hardware_destructive_interference_size) std::atomic<std::size_t> head_;
        alignas(hardware_destructive_interference_size) std::atomic<std::size_t> tail_;
    };

    // ── Logger ───────────────────────────────────────────────────────

    template<std::size_t NumWorkers = 1>
    class LoggerImpl {
        static_assert(NumWorkers >= 1 && NumWorkers <= 16);
    public:
        using Queue = std::conditional_t<NumWorkers == 1, std::deque<LogRecord>, LockFreeQueue<LogRecord>>;

        [[nodiscard]] static LoggerImpl& instance() { static LoggerImpl inst; return inst; }

        // ── API: 7 levels × 3 overloads ──────────────────────────────
        static void trace(std::string_view m, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Trace, m, {}, {}, l); }
        static void debug(std::string_view m, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Debug, m, {}, {}, l); }
        static void info(std::string_view m, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Info, m, {}, {}, l); }
        static void success(std::string_view m, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Success, m, {}, {}, l); }
        static void warning(std::string_view m, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Warning, m, {}, {}, l); }
        static void error(std::string_view m, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Error, m, {}, {}, l); }
        static void critical(std::string_view m, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Critical, m, {}, {}, l); }

        static void trace(std::string_view m, const ContextMap& c, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Trace, m, c, {}, l); }
        static void debug(std::string_view m, const ContextMap& c, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Debug, m, c, {}, l); }
        static void info(std::string_view m, const ContextMap& c, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Info, m, c, {}, l); }
        static void success(std::string_view m, const ContextMap& c, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Success, m, c, {}, l); }
        static void warning(std::string_view m, const ContextMap& c, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Warning, m, c, {}, l); }
        static void error(std::string_view m, const ContextMap& c, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Error, m, c, {}, l); }
        static void critical(std::string_view m, const ContextMap& c, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Critical, m, c, {}, l); }

        static void trace(std::string_view m, const ContextMap& c, std::string_view h, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Trace, m, c, h, l); }
        static void debug(std::string_view m, const ContextMap& c, std::string_view h, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Debug, m, c, h, l); }
        static void info(std::string_view m, const ContextMap& c, std::string_view h, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Info, m, c, h, l); }
        static void success(std::string_view m, const ContextMap& c, std::string_view h, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Success, m, c, h, l); }
        static void warning(std::string_view m, const ContextMap& c, std::string_view h, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Warning, m, c, h, l); }
        static void error(std::string_view m, const ContextMap& c, std::string_view h, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Error, m, c, h, l); }
        static void critical(std::string_view m, const ContextMap& c, std::string_view h, std::source_location l = std::source_location::current()) { instance().log(LogLevel::Critical, m, c, h, l); }

        // ── Handler management ───────────────────────────────────────
        [[nodiscard]] std::expected<void, ConfigError> add_handler(ConfigTemplate config) {
            std::unique_lock lock(handlers_mutex_);
            if (handlers_.size() >= 32)
                return std::unexpected(ConfigError::TooManyHandlers);
            handlers_.emplace_back(std::make_unique<Handler>(std::move(config)));
            return {};
        }

        void remove_handler(std::string_view name) {
            std::unique_lock lock(handlers_mutex_);
            std::erase_if(handlers_, [name](const auto& h) { return h->name() == name; });
        }

        void set_overflow_policy(OverflowPolicy p) noexcept { overflow_policy_.store(p); }

        struct Stats {
            std::size_t queued, dropped, processed, handlers;
        };

        [[nodiscard]] Stats get_stats() const noexcept {
            Stats s{};
            if constexpr (NumWorkers == 1) { std::lock_guard lock(queue_mutex_); s.queued = queue_.size(); }
            else s.queued = queue_.size();
            s.dropped = dropped_.load();
            s.processed = processed_.load();
            { std::shared_lock lock(handlers_mutex_); s.handlers = handlers_.size(); }
            return s;
        }

        void shutdown() {
            bool expected = true;
            if (!running_.compare_exchange_strong(expected, false))
                return;
            if constexpr (NumWorkers == 1) queue_cv_.notify_all();
            for (auto& w : workers_) if (w.joinable()) w.join();
        }

    private:
        LoggerImpl() : queue_([] -> Queue {
            if constexpr (NumWorkers == 1) return std::deque<LogRecord>{};
            else return LockFreeQueue<LogRecord>(8192);
            }()) {
            for (std::size_t i = 0; i < NumWorkers; ++i)
                workers_.emplace_back([this] { process_loop(); });
        }
        ~LoggerImpl() { shutdown(); }

        void log(LogLevel level, std::string_view msg, const ContextMap& ctx,
            std::string_view handler, std::source_location loc) {
            LogRecord rec(level, interpolate_message(msg, ctx), handler, loc);
            if (!ctx.empty()) {
                rec.context = ctx;
                if (msg.find("@{") != std::string_view::npos)
                    rec.lexeme = std::string(msg);
            }
            enqueue(std::move(rec));
        }

        void enqueue(LogRecord&& rec) {
            if constexpr (NumWorkers == 1) {
                std::unique_lock lock(queue_mutex_);
                if (queue_.size() >= kQueueCapacity) {
                    switch (overflow_policy_.load()) {
                    case OverflowPolicy::Block:
                        queue_cv_.wait(lock, [this] { return queue_.size() < kQueueCapacity || !running_.load(); });
                        break;
                    case OverflowPolicy::DropOldest: queue_.pop_front(); dropped_.fetch_add(1); break;
                    case OverflowPolicy::DropNewest: dropped_.fetch_add(1); return;
                    }
                }
                queue_.push_back(std::move(rec));
                queue_cv_.notify_one();
            }
            else {
                switch (overflow_policy_.load()) {
                case OverflowPolicy::Block:
                    while (running_.load()) {
                        if (queue_.try_enqueue(std::move(rec))) return;
                        std::this_thread::yield();
                    }
                    dropped_.fetch_add(1);
                    break;
                case OverflowPolicy::DropOldest: {
                    if (!queue_.try_enqueue(std::move(rec))) {
                        LogRecord old;
                        if (queue_.try_dequeue(old)) dropped_.fetch_add(1);
                        if (!queue_.try_enqueue(std::move(rec))) dropped_.fetch_add(1);
                    }
                    break;
                }
                case OverflowPolicy::DropNewest:
                    if (!queue_.try_enqueue(std::move(rec))) dropped_.fetch_add(1);
                    break;
                }
            }
        }

        void process_loop() {
            while (running_.load()) {
                LogRecord rec;
                bool got = false;
                if constexpr (NumWorkers == 1) {
                    std::unique_lock lock(queue_mutex_);
                    queue_cv_.wait(lock, [this] { return !queue_.empty() || !running_.load(); });
                    if (!running_.load() && queue_.empty()) break;
                    if (!queue_.empty()) {
                        rec = std::move(queue_.front()); queue_.pop_front();
                        got = true; queue_cv_.notify_one();
                    }
                }
                else {
                    got = queue_.try_dequeue(rec);
                    if (!got) { std::this_thread::sleep_for(std::chrono::microseconds(100)); continue; }
                }
                if (got) { dispatch(rec); processed_.fetch_add(1); }
            }
            // Drain remaining on shutdown
            if constexpr (NumWorkers == 1) {
                std::unique_lock lock(queue_mutex_);
                while (!queue_.empty()) {
                    auto r = std::move(queue_.front()); queue_.pop_front();
                    lock.unlock(); dispatch(r); processed_.fetch_add(1); lock.lock();
                }
            }
            else {
                LogRecord r;
                while (queue_.try_dequeue(r)) { dispatch(r); processed_.fetch_add(1); }
            }
        }

        void dispatch(const LogRecord& rec) {
            std::shared_lock lock(handlers_mutex_);
            for (auto& h : handlers_) if (h->accepts(rec)) h->emit(rec);
        }

        // ── Handler ──────────────────────────────────────────────────
        class Handler {
        public:
            explicit Handler(ConfigTemplate cfg)
                : name_(cfg.name()), min_level_(cfg.level()), filters_(cfg.filters()),
                formats_(cfg.formats()), outputs_(cfg.outputs()),
                base_ctx_(cfg.context()), structured_(cfg.is_structured()) {
            }

            [[nodiscard]] bool accepts(const LogRecord& r) const {
                if (r.level < min_level_) return false;
                if (!r.handler_hint.empty() && r.handler_hint != name_) return false;
                for (const auto& f : filters_) if (!f(r)) return false;
                return true;
            }

            void emit(const LogRecord& rec) {
                for (const auto& [fmt_name, target] : outputs_) {
                    auto it = formats_.find(fmt_name);
                    if (it == formats_.end()) continue;
                    try {
                        std::string out;
                        if (structured_) out = to_json(rec);
                        else { out = it->second.apply(rec, base_ctx_); out = ColorProcessor::process(out); out += '\n'; }
                        write(target, out);
                    }
                    catch (const std::exception& e) {
                        std::cerr << "[Logger] " << e.what() << '\n';
                    }
                }
            }

            [[nodiscard]] const std::string& name() const noexcept { return name_; }

        private:
            std::string to_json(const LogRecord& rec) const {
                std::ostringstream j;
                j << "{\"timestamp\":" << std::chrono::system_clock::to_time_t(rec.timestamp)
                    << ",\"level\":\"" << level_to_string(rec.level)
                    << "\",\"message\":\"" << json_escape(rec.message) << "\"";

                if (rec.lexeme)
                    j << ",\"message_template\":\"" << json_escape(*rec.lexeme) << "\"";

                j << ",\"thread\":\"" << rec.thread_id
                    << "\",\"file\":\"" << json_escape(rec.location.file_name())
                    << "\",\"line\":" << rec.location.line()
                    << ",\"function\":\"" << json_escape(rec.location.function_name()) << "\"";

                if (!rec.context.empty() || !base_ctx_.empty()) {
                    j << ",\"context\":{";
                    bool first = true;
                    auto emit = [&](const ContextMap& ctx) {
                        for (const auto& [k, v] : ctx) {
                            if (!first) j << ",";
                            first = false;
                            j << "\"" << json_escape(k) << "\":";
                            std::visit([&j](const auto& val) {
                                using T = std::decay_t<decltype(val)>;
                                if constexpr (std::is_same_v<T, bool>)
                                    j << (val ? "true" : "false");
                                else if constexpr (std::is_same_v<T, std::string>)
                                    j << "\"" << json_escape(val) << "\"";
                                else
                                    j << val;
                                }, v);
                        }
                        };
                    emit(base_ctx_);
                    emit(rec.context);
                    j << "}";
                }
                j << "}\n";
                return j.str();
            }

            void write(const ConfigTemplate::Output& target, const std::string& out) {
                std::visit([&out](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, StreamTarget>) {
                        if (auto* s = arg.get()) { *s << out; s->flush(); }
                    }
                    else if constexpr (std::is_same_v<T, std::filesystem::path>) {
                        if (auto r = get_file_cache().write(arg, out); !r)
                            std::cerr << "[Logger] " << arg << ": " << file_error_to_string(r.error()) << '\n';
                    }
                    }, target);
            }

            std::string name_;
            LogLevel min_level_;
            std::vector<ConfigTemplate::Filter> filters_;
            std::unordered_map<std::string, ParsedFormat> formats_;
            std::vector<std::pair<std::string, ConfigTemplate::Output>> outputs_;
            ContextMap base_ctx_;
            bool structured_;
        };

        static constexpr std::size_t kQueueCapacity = 8192;

        mutable std::mutex queue_mutex_;
        std::condition_variable queue_cv_;
        Queue queue_;
        std::atomic<OverflowPolicy> overflow_policy_{ OverflowPolicy::Block };

        mutable std::shared_mutex handlers_mutex_;
        std::vector<std::unique_ptr<Handler>> handlers_;

        std::atomic<std::size_t> dropped_{ 0 }, processed_{ 0 };
        std::atomic<bool> running_{ true };
        std::vector<std::thread> workers_;
    };

    using Logger = LoggerImpl<1>;
    template<std::size_t N> using MultiWorkerLogger = LoggerImpl<N>;

} // namespace Gem

// ── Macros ───────────────────────────────────────────────────────

#define LOG_TRACE(msg, ...)    ::Gem::Logger::trace(msg, ##__VA_ARGS__)
#define LOG_DEBUG(msg, ...)    ::Gem::Logger::debug(msg, ##__VA_ARGS__)
#define LOG_INFO(msg, ...)     ::Gem::Logger::info(msg, ##__VA_ARGS__)
#define LOG_SUCCESS(msg, ...)  ::Gem::Logger::success(msg, ##__VA_ARGS__)
#define LOG_WARNING(msg, ...)  ::Gem::Logger::warning(msg, ##__VA_ARGS__)
#define LOG_ERROR(msg, ...)    ::Gem::Logger::error(msg, ##__VA_ARGS__)
#define LOG_CRITICAL(msg, ...) ::Gem::Logger::critical(msg, ##__VA_ARGS__)
