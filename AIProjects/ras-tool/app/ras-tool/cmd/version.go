package cmd

import (
	"github.com/spf13/cobra"

	"qiniu.com/argus/AIProjects/ras-tool/com/version"
)

func init() {
	rootCmd.AddCommand(versionCmd)
}

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print version",
	Run: func(cmd *cobra.Command, args []string) {
		logger.Println("version:", version.Version())
	},
}
