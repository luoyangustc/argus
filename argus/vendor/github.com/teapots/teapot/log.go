package teapot

import (
	"fmt"
	"runtime"
	"strings"
)

type Level int

// RFC5424 log message levels.
const (
	LevelEmergency Level = iota + 1
	LevelAlert
	LevelCritical
	LevelError
	LevelWarning
	LevelNotice
	LevelInformational
	LevelDebug
)

var (
	DefaultLevelTag = [9]string{
		"[M]",
		"[A]",
		"[C]",
		"[E]",
		"[W]",
		"[N]",
		"[I]",
		"[D]",
	}

	DefaultColors = [9]brush{
		newBrush("37"), // white
		newBrush("36"), // cyan
		newBrush("35"), // magenta
		newBrush("31"), // red
		newBrush("33"), // yellow
		newBrush("32"), // green
		newBrush("34"), // blue
		newBrush("30"), // gray
	}
)

type LineOpt struct {
	Depth int
	Stack bool
}

type LogPrinter interface {
	Print(...interface{})
}

type LoggerAdv interface {
	Logger

	SetPrefix(string)
	SetPrefixParser(prefixParser PrefixParser)
	SetLevel(Level)
	SetColorMode(bool)
	SetFuncDepth(int)
	SetLineInfo(bool)
	SetShortLine(bool)
	SetFlatLine(bool)
	EnableLogStack(Level)
}

type Logger interface {
	Emergencyf(string, ...interface{})
	Emergency(...interface{})
	Alertf(string, ...interface{})
	Alert(...interface{})
	Critcialf(string, ...interface{})
	Critcial(...interface{})
	Errorf(string, ...interface{})
	Error(...interface{})
	Warnf(string, ...interface{})
	Warn(...interface{})
	Noticef(string, ...interface{})
	Notice(...interface{})
	Infof(string, ...interface{})
	Info(...interface{})
	Debugf(string, ...interface{})
	Debug(...interface{})
	Recover(func(), ...Level) interface{}
}

type PrefixParser func([]interface{}) (string, []interface{})

type native interface {
	funcCallMsg(depth int) string
	print(Level, string, string, LineOpt)
}

type baseLogger struct {
	out LogPrinter

	native native

	prefix string
	level  Level
	stack  Level
	color  bool
	line   bool
	short  bool
	depth  int
	flat   bool

	prefixParser PrefixParser
}

var _ Logger = new(baseLogger)

func NewLogger(out LogPrinter) LoggerAdv {
	l := &baseLogger{
		out:   out,
		level: LevelDebug,
		depth: 4,
		flat:  true,
	}
	l.native = l
	return l
}

func (l *baseLogger) SetPrefix(prefix string) {
	l.prefix = prefix
}

func (l *baseLogger) SetPrefixParser(prefixParser PrefixParser) {
	l.prefixParser = prefixParser
}

func (l *baseLogger) SetLevel(level Level) {
	l.level = level
}

func (l *baseLogger) SetColorMode(color bool) {
	l.color = color
}

func (l *baseLogger) SetFuncDepth(depth int) {
	l.depth = depth
}

func (l *baseLogger) SetLineInfo(show bool) {
	l.line = show
}

func (l *baseLogger) SetShortLine(short bool) {
	l.short = short
}

func (l *baseLogger) SetFlatLine(flat bool) {
	l.flat = flat
}

func (l *baseLogger) EnableLogStack(level Level) {
	l.stack = level
}

func (l *baseLogger) Emergencyf(format string, v ...interface{}) {
	l.writef(LevelEmergency, format, v...)
}

func (l *baseLogger) Emergency(v ...interface{}) {
	l.write(LevelEmergency, v...)
}

func (l *baseLogger) Alertf(format string, v ...interface{}) {
	l.writef(LevelAlert, format, v...)
}

func (l *baseLogger) Alert(v ...interface{}) {
	l.write(LevelAlert, v...)
}

func (l *baseLogger) Critcialf(format string, v ...interface{}) {
	l.writef(LevelCritical, format, v...)
}

func (l *baseLogger) Critcial(v ...interface{}) {
	l.write(LevelCritical, v...)
}

func (l *baseLogger) Errorf(format string, v ...interface{}) {
	l.writef(LevelError, format, v...)
}

func (l *baseLogger) Error(v ...interface{}) {
	l.write(LevelError, v...)
}

func (l *baseLogger) Warnf(format string, v ...interface{}) {
	l.writef(LevelWarning, format, v...)
}

func (l *baseLogger) Warn(v ...interface{}) {
	l.write(LevelWarning, v...)
}

func (l *baseLogger) Noticef(format string, v ...interface{}) {
	l.writef(LevelNotice, format, v...)
}

func (l *baseLogger) Notice(v ...interface{}) {
	l.write(LevelNotice, v...)
}

func (l *baseLogger) Infof(format string, v ...interface{}) {
	l.writef(LevelInformational, format, v...)
}

func (l *baseLogger) Info(v ...interface{}) {
	l.write(LevelInformational, v...)
}

func (l *baseLogger) Debugf(format string, v ...interface{}) {
	l.writef(LevelDebug, format, v...)
}

func (l *baseLogger) Debug(v ...interface{}) {
	l.write(LevelDebug, v...)
}

func (l *baseLogger) Recover(f func(), level ...Level) (err interface{}) {
	defer func() {
		if err = recover(); err != nil {
			if len(level) > 0 {
				l.write(level[0], err, LineOpt{Stack: true, Depth: 4})
			} else {
				l.Error(err, LineOpt{Stack: true, Depth: 5})
			}
		}
	}()

	// call user function
	f()
	return nil
}

func (l *baseLogger) parseValues(values []interface{}) (v []interface{}, opt LineOpt) {
	opt.Depth = l.depth
	v = values
	if len(v) > 0 {
		if o, ok := v[len(v)-1].(LineOpt); ok {
			v = v[:len(v)-1]
			opt = o
			opt.Depth += l.depth
		}
	}
	return
}

func (l *baseLogger) writef(level Level, format string, v ...interface{}) {
	if level > l.level {
		return
	}

	v, opt := l.parseValues(v)

	prefix, v := l.getPrefix(v)

	msg := fmt.Sprintf(format, v...)
	l.native.print(level, prefix, msg, opt)
}

func (l *baseLogger) write(level Level, v ...interface{}) {
	if level > l.level {
		return
	}

	v, opt := l.parseValues(v)

	prefix, v := l.getPrefix(v)

	msg := fmt.Sprintln(v...)
	l.native.print(level, prefix, msg, opt)
}

func (l *baseLogger) print(level Level, prefix, msg string, opt LineOpt) {
	out := DefaultLevelTag[level-1] + prefix

	if l.line {
		out += l.native.funcCallMsg(opt.Depth)
	}

	if l.color {
		out = DefaultColors[level-1](out)
	}

	out = out + " " + msg

	if (l.stack >= level || opt.Stack) &&

		// skip log stack with prefix PANIC / STACK
		!(len(msg) > 5 && (msg[:5] == "PANIC" || msg[:5] == "STACK")) {

		out += "\n" + string(Stack(opt.Depth))
	}

	if l.flat {
		out = strings.TrimSuffix(out, "\n")
		out = strings.Replace(out, "\n", "\\n", -1)
	}

	l.out.Print(out)
}

func (l *baseLogger) getPrefix(v []interface{}) (string, []interface{}) {
	prefix := l.prefix
	if l.prefixParser != nil {
		var p string
		p, v = l.prefixParser(v)
		prefix += p
	}
	return prefix, v
}

func (l *baseLogger) funcCallMsg(depth int) string {
	if depth > 0 {
		_, file, line, ok := runtime.Caller(depth)
		if ok {
			if l.short {
				parts := strings.SplitN(file, `/src/`, 2)
				if len(parts) > 0 {
					file = parts[len(parts)-1]
				}
			}
			return fmt.Sprintf(`["%s:%d"]`, file, line)
		}
	}
	return ""
}

type ReqLogger interface {
	Logger
	ReqId() string
}

type ReqLoggerAdv interface {
	LoggerAdv
	ReqId() string
}

type reqLogger struct {
	*baseLogger

	reqId     string
	reqPrefix string
}

// TODO remove headerKey
func NewReqLogger(out LogPrinter, headerKey string, reqId string) ReqLoggerAdv {

	log := &reqLogger{
		baseLogger: NewLogger(out).(*baseLogger),
	}
	log.native = log
	log.reqId = reqId
	log.reqPrefix = "[" + log.reqId + "]"
	log.prefix = log.reqPrefix

	return log
}

func (l *reqLogger) SetPrefix(prefix string) {
	l.prefix = prefix + l.reqPrefix
}

func (l *reqLogger) ReqId() string {
	return l.reqId
}

func NewWithId(out LogPrinter, id string) ReqLoggerAdv {
	log := &reqLogger{
		baseLogger: NewLogger(out).(*baseLogger),
	}
	log.native = log

	log.reqId = id

	log.reqPrefix = "[" + id + "]"
	log.prefix = log.reqPrefix

	return log
}
