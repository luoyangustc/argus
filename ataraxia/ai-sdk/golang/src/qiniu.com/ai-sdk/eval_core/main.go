package main

import (
	"context"
	"os"

	"github.com/qiniu/x/xlog.v7"
	"github.com/spf13/cobra"
	"qbox.us/cc/config"
	"qiniupkg.com/x/log.v7"
)

var xl = xlog.New("main")

func ce(err error) {
	if err != nil {
		panic(err)
	}
}

var rootCmd = &cobra.Command{
	Use: "eval_core",
	RunE: func(cmd *cobra.Command, args []string) error {
		return cmd.Help()
	},
}
var daemonCmd = &cobra.Command{
	Use: "daemon",
	RunE: func(cmd *cobra.Command, args []string) error {
		// TODO: 支持指定配置文件
		// TODO: root context
		// TODO: 日志、监控转发
		// 加载配置
		{
			if err := config.LoadEx(&cfg, cfg.Conf); err != nil {
				xl.Fatal("Failed to load configure file!")
			}
		}
		ctx := context.Background()
		log.SetOutputLevel(cfg.DebugLevel)
		go runBatchMq(ctx)
		go runMonitor(ctx, MONITOR_HTTP_ADDR)
		go runGrpcServer(ctx, INFERENCE_GRPC_ADDR)
		runCmds()
		return nil
	},
}

var mqCmd = &cobra.Command{
	Use: "mq",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := context.Background()
		go runMonitor(ctx, MONITOR_HTTP_ADDR)
		runBatchMq(ctx)
		return nil
	},
}

type Process struct {
	Args       []string `json:"args"`
	Cmd        string   `json:"cmd"`
	Dir        string   `json:"dir"`
	Env        []string `json:"env"`
	Name       string   `json:"name"`
	WithSysEnv bool     `json:"with_sys_env"`
}

var cfg struct {
	DebugLevel int       `json:"debug_level"`
	App        string    `json:"app"`
	Process    []Process `json:"process"`
	Conf       string    `json:"conf"`
}

func init() {
	daemonCmd.Flags().StringVarP(&cfg.Conf, "conf", "f", "", "conf file")
	rootCmd.AddCommand(daemonCmd, mqCmd)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		xl.Error(err)
		os.Exit(1)
	}
}
