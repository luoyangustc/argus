package conf

type BatchEntryProcessorConf struct {
	MaxPool     int  `json:"pool_max"`      //BatchEntryJobProcessor处理job启动的goroutine数
	Gzip        bool `json:"gzip"`          //机审的结果文件是否为zip文件
	MaxFileLine int  `json:"max_file_line"` //每个结果文件最多保存的行数
	MaxCapTask  int  `json:"max_cap_task"`  //发给CAP的tasks个数（即满了多少个就把task发送给cap)
}

type BatchEntryResultConf struct {
	MaxPool    int `json:"pool_max"`   //BatchEntryJobResult处理job启动的goroutine数
	ChecktTime int `json:"check_time"` //检查所有正在被处理的job是否完成的间隔时间，以秒为单位
}
