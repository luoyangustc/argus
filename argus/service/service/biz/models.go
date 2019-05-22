package biz

type EvalRunType string

const (
	EvalRunTypeSDK     EvalRunType = "sdk"
	EvalRunTypeServing EvalRunType = "serving-eval"
)

type EvalModelConfig struct {
	Image       string            `yaml:"image" json:"image"`
	Model       string            `yaml:"model,omitempty" json:"model,omitempty"`
	CustomFiles map[string]string `yaml:"custom_files,omitempty" json:"custom_files,omitempty"`
	Args        *ModelConfigArgs  `yaml:"args,omitempty" json:"args,omitempty"`
	Type        EvalRunType       `yaml:"type,omitempty" json:"type,omitempty"`
}

type ModelConfigArgs struct {
	BatchSize    int         `yaml:"batch_size,omitempty" json:"batch_size,omitempty"`
	ImageWidth   int         `yaml:"image_width,omitempty" json:"image_width,omitempty"`
	CustomValues interface{} `yaml:"custom_values,omitempty" json:"custom_values,omitempty"`
}
