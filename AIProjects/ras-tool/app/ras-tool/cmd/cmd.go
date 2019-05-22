package cmd

import (
	"log"
	"os"

	"github.com/spf13/cobra"
)

var (
	logger *log.Logger
)

var rootCmd = cobra.Command{
	Use:   "ras-tool",
	Short: "ras-tool is a tool for deploying and managing AI projects.",
	RunE: func(cmd *cobra.Command, args []string) error {
		return cmd.Help()
	},
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		logger.Println(err)
		os.Exit(-1)
	}
}

func init() {
	logger = log.New(os.Stdout, "", 0)
}
