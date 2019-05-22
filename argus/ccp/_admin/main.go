package main

import "github.com/spf13/cobra"

func main() {

	_ = AdminConfig.Load("config.json")

	var rootCmd = &cobra.Command{Use: "shell", Args: cobra.MinimumNArgs(1)}
	rootCmd.AddCommand(PfopRuleCmd())
	rootCmd.AddCommand(CcpRuleCmd())
	rootCmd.AddCommand(BucketCmd())
	rootCmd.AddCommand(CcpReviewCmd())

	rootCmd.Execute()
}
