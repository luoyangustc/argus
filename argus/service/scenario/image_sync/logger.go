// 暂不用
package image_sync

// import (
// 	"bufio"
// 	"context"
// 	"errors"
// 	"fmt"
// 	"os"
// 	"time"

// 	"github.com/go-kit/kit/endpoint"
// 	kitlog "github.com/go-kit/kit/log"

// 	xlog "github.com/qiniu/xlog.v1"

// 	"qiniu.com/argus/service/middleware"
// )

// var (
// 	defaultLogWriter = &logWriter{}
// )

// const (
// 	xlogKey key = 0
// )

// type key int
// type logConfig struct {
// 	LogDir  string `json:"log_dir"`
// 	MaxSize int    `json:"max_size"`
// }
// type ImgLogger struct {
// 	kitlog.Logger
// }
// type logWriter struct {
// 	logConfig
// 	Size int
// 	Num  int
// 	*bufio.Writer
// 	*os.File
// }

// func (lg *logWriter) Write(p []byte) (n int, err error) {
// 	_, err = lg.Writer.Write(p)
// 	if err != nil {
// 		return
// 	}
// 	lg.Size += len(p)
// 	if lg.Size >= lg.MaxSize {
// 		_ = lg.Writer.Flush()
// 		_ = lg.File.Close()
// 		lg.Num++
// 		fileName := lg.LogDir + "/log_" + fmt.Sprintf("%d_%d", time.Now().UnixNano(), defaultLogWriter.Num)
// 		lg.File, err = os.OpenFile(fileName, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0744)
// 		if err != nil {
// 			return
// 		}
// 		lg.Size = 0
// 		lg.Writer.Reset(lg.File)
// 	}
// 	return
// }
// func InitLogger(conf logConfig) error {
// 	if conf.LogDir == "" {
// 		return errors.New("log dir is required")
// 	}
// 	if conf.MaxSize == 0 {
// 		conf.MaxSize = 64 * 1024 * 1024
// 	}
// 	fileName := conf.LogDir + "/log_" + fmt.Sprintf("%d_%d", time.Now().UnixNano(), defaultLogWriter.Num)
// 	file, err := os.OpenFile(fileName, os.O_CREATE|os.O_EXCL|os.O_RDWR, 0744)
// 	if err != nil {
// 		return err
// 	}
// 	defaultLogWriter = &logWriter{
// 		logConfig: conf,
// 		Writer:    bufio.NewWriter(file),
// 		File:      file,
// 	}
// 	return nil
// }
// func NewLogger(keyvals ...interface{}) *ImgLogger {
// 	var logger kitlog.Logger
// 	{
// 		if defaultLogWriter.Writer == nil {
// 			logger = kitlog.NewLogfmtLogger(os.Stderr)
// 		} else {
// 			logger = kitlog.NewLogfmtLogger(kitlog.NewSyncWriter(defaultLogWriter))
// 		}
// 		logger = kitlog.With(logger, "ts", kitlog.DefaultTimestampUTC)
// 		logger = kitlog.With(logger, "caller", kitlog.DefaultCaller)
// 		logger = kitlog.With(logger, keyvals...)
// 	}
// 	return &ImgLogger{
// 		Logger: logger,
// 	}
// }
// func FromContext(ctx context.Context) *ImgLogger {
// 	il, ok := ctx.Value(xlogKey).(*ImgLogger)
// 	if !ok {
// 		il = NewLogger()
// 	}
// 	return il
// }
// func (m ImgLogger) New(svc middleware.Service, endpoints middleware.ServiceEndpoints) (middleware.Service, error) {
// 	return middleware.MakeMiddlewareFactory(nil,
// 		func(methodName string, service middleware.Service, defaultEndpoint func() endpoint.Endpoint) endpoint.Endpoint {
// 			e := defaultEndpoint()
// 			return func(ctx context.Context, request interface{}) (response interface{}, err error) {
// 				reqid := xlog.GenReqId()
// 				defer func(begin time.Time) {
// 					_ = m.Log("req", reqid, "method", methodName, "took", time.Since(begin), "err", err)
// 				}(time.Now())
// 				ctex := context.WithValue(ctx, xlogKey, NewLogger("reqid", reqid))
// 				return e(ctex, request)
// 			}
// 		},
// 	)(svc, endpoints)
// }
