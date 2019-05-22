package config

import (
	"bytes"
	"encoding/json"
	"flag"
	"io/ioutil"
)

import (
	"github.com/qiniu/log.v1"
	"qbox.us/cc"
)

var (
	confName *string
	NL       = []byte{'\n'}
	ANT      = []byte{'#'}
)

func Init(cflag, app, default_conf string) {

	confDir, _ := cc.GetConfigDir(app)
	confName = flag.String(cflag, confDir+"/"+default_conf, "the config file")
}

func ConfName() string {
	if confName != nil {
		return *confName
	}
	return ""
}

func Load(conf interface{}) (err error) {

	if !flag.Parsed() {
		flag.Parse()
	}

	log.Info("Use the config file of ", *confName)
	return LoadEx(conf, *confName)
}

func trimComments(data []byte) (data1 []byte) {

	conflines := bytes.Split(data, NL)
	for k, line := range conflines {
		conflines[k] = trimCommentsLine(line)
	}
	return bytes.Join(conflines, NL)
}

func trimCommentsLine(line []byte) []byte {

	var newLine []byte
	var i, quoteCount int
	lastIdx := len(line) - 1
	for i = 0; i <= lastIdx; i++ {
		if line[i] == '\\' {
			if i != lastIdx && (line[i+1] == '\\' || line[i+1] == '"') {
				newLine = append(newLine, line[i], line[i+1])
				i++
				continue
			}
		}
		if line[i] == '"' {
			quoteCount++
		}
		if line[i] == '#' {
			if quoteCount%2 == 0 {
				break
			}
		}
		newLine = append(newLine, line[i])
	}
	return newLine
}

func LoadEx(conf interface{}, confName string) (err error) {

	data, err := ioutil.ReadFile(confName)
	if err != nil {
		log.Error("Load conf failed:", err)
		return
	}
	data = trimComments(data)

	err = json.Unmarshal(data, conf)
	if err != nil {
		log.Error("Parse conf failed:", err)
	}
	return
}

func LoadFile(conf interface{}, confName string) (err error) {

	data, err := ioutil.ReadFile(confName)
	if err != nil {
		return
	}
	data = trimComments(data)

	return json.Unmarshal(data, conf)
}

func LoadData(conf interface{}, data []byte) (err error) {
	data = trimComments(data)

	err = json.Unmarshal(data, conf)
	if err != nil {
		log.Error("Parse conf failed:", err)
	}
	return
}
