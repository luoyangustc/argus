package configs

import (
	"fmt"
	"os"

	"github.com/BurntSushi/toml"
	"qiniu.com/argus/test/lib/auth"
)

// Infos from config file
type Config struct {
	Users                    map[string]user `toml:"users"`
	Host                     host
	Atservingprivatebucketz0 userBucket `toml:"privatebucket"`
	Publicbucket             userBucket
	ArgusBcpTestbucket       userBucket
}

type StubConfig struct {
	Host    host
	Servers server `toml:"servers"`
}
type online struct {
	ImagePulp       bool `toml:"imagepulp"`
	ImagePolitician bool `toml:"imagepolitician"`
	ImageTerror     bool `toml:"imageterror"`
	ImageAds        bool `toml:"imageads"`
}
type server struct {
	Online     online                    `toml:"online"`
	VideoAsync bool                      `toml:"videoasync"`
	Type       map[string]map[string]app `toml:"type"`
}

type app struct {
	Op        string   `toml:"op"`
	Tsv       string   `toml:"tsv"`
	Tsvs      []string `toml:"tsvs"`
	Set       string   `toml:"set"`
	Sets      []string `toml:"sets"`
	Precision float64  `toml:"precision"`
	Path      string   `toml:"path"`
	EvalPath  string   `toml:"evalpath"`
	Limit     int      `toml:"limit"`
	Version   string   `toml:"version"`
}

// service host
type host struct {
	ACC                 string
	RS                  string `toml:"rs"` // 实际映射到的是zoneproxy的地址
	RSF                 string
	API                 string
	UC                  string
	DOC                 string
	Counter             string
	ONE                 string
	UFOP                string
	IO                  string
	UP                  string `toml:"up"`
	Apigate_up          string
	PILI                string `toml:"piliapi"`
	PILIV2              string `toml:"piliapiv2"`
	TBLMGR              string //内网访问
	PFDTRACKER          string //内网访问
	PFDSTG              string //内网访问
	PTFDSTG             string //内网访问
	CONFG               string //内网访问
	PFDCFG              string
	RS_INTERAL          string `toml:"rs_interal"` // rs的内网地址
	ZONEPROXY           string
	QINIUPROXY          string `toml:"qiniuproxy"`
	MC                  string
	KMQ                 string
	PANDORA             string
	HUBCLOUD            string `toml:"hubcloud"`
	VANCE               string `toml:"vance"`
	AT_SERVING          string
	AT_ARGUS            string
	AT_NET              string
	ARGUS_VIDEO         string
	OLD_ARGUS           string
	ARGUS_GROUP         string `toml:"argus_group"`
	ARGUS_BJOB          string `toml:"argus_bjob"`
	ARGUS_BCP           string
	AT_SERVING_GATE     string `toml:"at_serving_gate"`
	AT_ARGUS_GATE       string `tomal:"at_argus_gate"`
	AT_CENSOR_GATE      string `toml:"at_censor_gate"`
	AT_CCP_MANAGER_GATE string `toml:"at_ccp_manager_gate"`
	SOURCE              string `tomal:"source"`
	QBOX                string `tomal:"qbox"`
	FEATURE_GROUP_CPU   string `tomal:"feature_group_cpu"`
	FEATURE_GROUP_GPU   string `tomal:"feature_group_gpu"`
	AT_LIVE_VIDO        string `toml:"at_live_vido"`
	AT_LIVE_FACE        string `toml:"at_live_face"`
	AT_CCP_MANUAL_GATE  string `tomal:"at_ccp_manual_gate"`
	AT_CCP_REVIEW_GATE  string `tomal:"at_ccp_review_gate"`
	AT_CAP_ADMIN_GATE   string `tomal:"at_cap_admin_gate"`
	ARGUS_DBSTORAGE     string `toml:"argus_dbstorage"`
}

// user for original test
type user struct {
	Username string
	Password string
	AK       string `toml:"accesskey"`
	SK       string `toml:"secretkey"`
	Uid      uint32
}

type userBucket struct {
	Username string
	Name     string
	Domain   string
	User     user
}

//load configfile by env
func selectConfigFile() string {
	env := os.Getenv("TEST_ENV")
	if env == "" {
		panic("Please set environment: TEST_ENV")
	}

	zone := os.Getenv("TEST_ZONE")
	if zone == "" {
		panic("Please set environment: TEST_ZONE")
	}
	return env + "." + zone + ".conf"
}

// Reads info from config file
func ReadConfig() Config {
	var configfile string
	var config Config
	if os.Getenv("TEST_ENV") == "" {
		return config
	} else {
		configfile = os.Getenv("GOPATH") + "/src/qiniu.com/argus/test/configs/bucket.conf"
	}
	fmt.Printf("CHOOSE CONFIG FILE: %v \n", configfile)
	if _, err := toml.DecodeFile(configfile, &config); err != nil {
		panic("Read Config file failed: " + err.Error())
	}
	return config
}

func StubReadConfig() StubConfig {
	var configfile string
	if os.Getenv("TEST_ENV") == "" {
		// configfile = os.Getenv("GOPATH") + "/src/qiniu.com/argus/test/configs/locale.z0.conf"
		configfile = "locale.z0.conf"
	} else {
		configfile = os.Getenv("GOPATH") + "/src/qiniu.com/argus/test/configs/" + selectConfigFile()
	}
	fmt.Printf("CHOOSE CONFIG FILE: %v \n", configfile)
	var stubconfig StubConfig
	if _, err := toml.DecodeFile(configfile, &stubconfig); err != nil {
		panic("Read Config file failed: " + err.Error())
	}
	return stubconfig
}

var Configs = func() Config {
	config := ReadConfig()

	config.Atservingprivatebucketz0.User = config.Users[config.Atservingprivatebucketz0.Username]
	config.Publicbucket.User = config.Users[config.Publicbucket.Username]
	config.ArgusBcpTestbucket.User = config.Users[config.ArgusBcpTestbucket.Username]

	return config
}()

var StubConfigs = func() StubConfig {
	stubconfig := StubReadConfig()
	return stubconfig
}()

var GeneralUser = auth.AccessInfo{Key: Configs.Users["general"].AK, Secret: Configs.Users["general"].SK}
var GeneralUseronline = auth.AccessInfo{Key: Configs.Users["generalonline"].AK, Secret: Configs.Users["generalonline"].SK}
var CsBucketUser = auth.AccessInfo{Key: Configs.Users["argusbcptestbucket"].AK, Secret: Configs.Users["argusbcptestbucket"].SK}
