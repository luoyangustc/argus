package shell

import "github.com/spf13/cobra"

var RootCmd = &cobra.Command{
	Use: "tuso-cli",
	RunE: func(cmd *cobra.Command, args []string) error {
		return cmd.Help()
	},
}

var batchCmd = &cobra.Command{
	Use: "batch",
	RunE: func(cmd *cobra.Command, args []string) error {
		return cmd.Help()
	},
}

var hubCmd = &cobra.Command{
	Use: "hub",
	RunE: func(cmd *cobra.Command, args []string) error {
		return cmd.Help()
	},
}

func init() {
	RootCmd.AddCommand(batchCmd)
	RootCmd.AddCommand(hubCmd)

	batchCmd.AddCommand(batchAddCmd)

	hubCmd.AddCommand(hubCreateCmd)
	hubCmd.AddCommand(hubListCmd)
	hubCmd.AddCommand(hubInfoCmd)

	RootCmd.AddCommand(submitJob)
}
