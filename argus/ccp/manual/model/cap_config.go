package model

import "time"

type CAPConfig struct {
	Host string `json:"host"`
	//CallbackURL string        `json:"callback_url"` //cap发送结果给ccp/manual的地址
	Timeout time.Duration `json:"timeout"`
}
