package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"html/template"
	"io/ioutil"
	"strconv"
	"strings"
	"time"

	etcd "github.com/coreos/etcd/clientv3"
	"github.com/docopt/docopt-go"
	yaml "gopkg.in/yaml.v2"

	"github.com/pkg/errors"
	"github.com/qiniu/log.v1"
	"github.com/qiniu/rpc.v3"
	"qiniu.com/auth/qiniumac.v1"
	doraController "qiniu.com/dora/controller/api"
)

type deployConfig struct {
	Admin struct {
		AK string `json:"ak"`
		SK string `json:"sk"`
	} `json:"admin"`
	DoraController doraController.Config `json:"dora_controller"`
	Etcd           struct {
		Hosts []string `json:"hosts"`
	} `json:"etcd"`
	Default struct {
	} `json:"default"`
}

type appConfig struct {
	Release doraController.ReleaseArgs `json:"release" yaml:"release"`
	Config  struct {
		Meta    interface{} `json:"meta,omitempty" yaml:"meta,omitempty"`
		Release interface{} `json:"release,omitempty" yaml:"release,omitempty"`
	} `json:"config"`
}

type appRelease struct {
	Version string    `json:"version"`
	Image   string    `json:"image"`
	Flavor  string    `json:"flavor"`
	Ctime   time.Time `json:"ctime"`

	Expect uint `json:"expect"`
	Actual uint `json:"actual"`
}

func deployMain(args []string) {

	var usage = `deploy eval app. 

Usage:
  deploy -f <config.json> app <app>
  deploy -f <config.json> release <app> -i <image> [-m <model>] [-v <version>] [-d <desc>] [--config <app.yaml>]
  deploy -f <config.json> release <app> -i <image> [--copy <copyVersion>] [-v <version>] [-d <desc>] [--config <app.yaml>]
  deploy -f <config.json> resize <app> <release> <size>
  deploy -f <config.json> ps
  deploy -f <config.json> ps -a

Options:
  -h --help		Show this screen.
  --version		Show version.
  -v			App Release Version.
  -d			App Release desc.
  -m			Model URL.
  --copy		Copy From Release.
  --config		Config File.

`

	fmt.Printf("%#v\n", args)
	arguments, err := docopt.Parse(usage, args, true, "Deploy 0.2.0", false)
	if err != nil {
		log.Fatalf("parse args failed. %v %v\n%s", args, err, usage)
	}

	var (
		getStringArg = func(argm map[string]interface{}, key, defaultValue string) string {
			if v := argm[key]; v != nil {
				return v.(string)
			}
			return defaultValue
		}

		configFile      = getStringArg(arguments, "<config.json>", "")
		version         = getStringArg(arguments, "-v", "")
		image           = getStringArg(arguments, "<image>", "")
		model           = getStringArg(arguments, "-m", "")
		copyFromVersion = getStringArg(arguments, "--copy", "")
		desc            = getStringArg(arguments, "-d", "")
		appConfigFile   = getStringArg(arguments, "--config", "")

		config    deployConfig
		appConfig appConfig

		dc *doraController.Client
	)

	log.Infof("args: %#v", arguments)

	if err := readJSON(configFile, &config); err != nil {
		log.Fatalf("read config file failed. %s %v", configFile, err)
	}

	dc = doraController.New(config.DoraController)

	switch {
	case arguments["app"].(bool):
		var (
			ctx = context.Background()
			app = getStringArg(arguments, "<app>", "")
		)
		err := dc.CreateUfop(ctx, doraController.UfopArgs{Name: app, Mode: "private"})
		if err != nil {
			log.Fatalf("create app failed. %s %v", app, err)
		}
		if err = setUfopOfficial(ctx, app,
			doraController.Config{
				AccessKey: config.Admin.AK,
				SecretKey: config.Admin.SK,
				Host:      config.DoraController.Host,
			},
		); err != nil {
			log.Fatalf("set app official failed. %s %v", app, err)
		}
		log.Infof("create app done. %s", app)
	case arguments["release"].(bool):
		var (
			ctx = context.Background()
			app = getStringArg(arguments, "<app>", "")

			etcdc, err = etcd.New(
				etcd.Config{
					Endpoints:   config.Etcd.Hosts,
					DialTimeout: time.Second,
				},
			)

			etcdMetadata = func(app string) string {
				return fmt.Sprintf("/ava/serving/app/metadata/%s", strings.TrimPrefix(app, "ava-"))
			}
			etcdRelease = func(app, version string) string {
				return fmt.Sprintf("/ava/serving/app/release/%s/%s", strings.TrimPrefix(app, "ava-"), version)
			}
		)
		if err != nil {
			log.Fatalf("new etcd failed. %v", err)
		}
		appConfig.Config.Meta = new(interface{})
		appConfig.Config.Release = new(interface{})
		if copyFromVersion != "" {
			srcRelease, err := dc.GetReleaseInfoBrief(ctx, app, copyFromVersion)
			if err != nil {
				log.Fatalf("get copy release info failed. %s %v", copyFromVersion, err)
			}
			appConfig.Release = doraController.ReleaseArgs{
				Desc:   srcRelease.Desc,
				Image:  srcRelease.Image,
				Flavor: srcRelease.Flavor,
				// MntPath:
				// Port:
				HealthCk:     srcRelease.HealthCk,
				Env:          srcRelease.Env,
				LogFilePaths: srcRelease.LogFilePaths,
			}
			log.Infof("copy release. %#v", appConfig.Release)
			{
				if err := etcdGet(
					ctx, etcdc,
					etcdRelease(app, copyFromVersion),
					appConfig.Config.Release,
				); err != nil {
					log.Fatalf(
						"get copy release config failed. %s %v",
						etcdRelease(app, copyFromVersion), err,
					)
				}
				log.Infof("copy etcd release. %#v", appConfig.Config.Release)
			}
		}
		if appConfigFile != "" {
			if err := readYAML(appConfigFile, &appConfig); err != nil {
				log.Fatalf("read app config file failed. %s %v", appConfigFile, err)
			}
			appConfig.Config.Meta = cleanupMapValue(appConfig.Config.Meta)
			appConfig.Config.Release = cleanupMapValue(appConfig.Config.Release)
		}
		if version == "" {
			version = time.Now().Format("200601021504")
		}
		log.Infof("app version: %s", version)
		appConfig.Release.Verstr = version
		if desc != "" {
			appConfig.Release.Desc = desc
		}
		if image != "" {
			appConfig.Release.Image = image
		}
		if model != "" {
			bs, _ := json.Marshal(appConfig.Config.Release)
			buf := bytes.NewBuffer(nil)
			if err := template.
				Must(template.New("release").Parse(string(bs))).
				Execute(buf, struct{ Model string }{Model: model}); err != nil {
				log.Fatalf("set model failed. %v", err)
			}
			json.Unmarshal(buf.Bytes(), appConfig.Config.Release)
		}
		log.Infof("app config. %#v", appConfig)

		if isNil(appConfig.Config.Meta) == false {
			resp, err := etcd.NewKV(etcdc).Get(ctx, etcdMetadata(app))
			if err != nil {
				log.Fatalf("get etcd metadata failed. %s %v", etcdMetadata(app), err)
			}
			if resp.Kvs == nil || len(resp.Kvs) == 0 {
				if err := etcdPut(
					ctx, etcdc,
					etcdMetadata(app),
					appConfig.Config.Meta,
				); err != nil {
					log.Fatalf("put etcd metadata failed. %v", err)
				}
			}
		}
		if isNil(appConfig.Config.Release) == false {
			resp, err := etcd.NewKV(etcdc).Get(ctx, etcdRelease(app, version))
			if err != nil {
				log.Fatalf("get etcd release failed. %s %v", etcdRelease(app, version), err)
			}
			if resp.Kvs == nil || len(resp.Kvs) == 0 {
				if err := etcdPut(
					ctx, etcdc,
					etcdRelease(app, version),
					appConfig.Config.Release,
				); err != nil {
					log.Fatalf("put etcd release failed. %v", err)
				}
			}
		}

		ctx2, cancel := context.WithTimeout(ctx, time.Second*5)
		defer cancel()
		if err := dc.CreateRelease(ctx2, app, appConfig.Release); err != nil {
			log.Fatalf("create release failed. %v", err)
		}
		log.Infof("CreateRelease %s %s success", app, version)
	case arguments["resize"].(bool):
		var (
			ctx     = context.Background()
			app     = getStringArg(arguments, "<app>", "")
			release = getStringArg(arguments, "<release>", "")
			size, _ = strconv.ParseUint(getStringArg(arguments, "<size>", "0"), 10, 64)
		)
		resp, err := dc.CreateDeployment(ctx, app,
			doraController.DeploymentArgs{
				Verstr: release,
				Region: "z0",
				Expect: uint(size),
			},
		)
		if err != nil {
			log.Fatalf("resize app failed. %s %s %d %v", app, release, size, err)
		}
		log.Infof("try resize app... %s %s %d %s", app, release, size, resp.DeploymentID)
	case arguments["ps"].(bool):
		var (
			ctx     = context.Background()
			apps    = map[string][]appRelease{}
			showALL = arguments["-a"].(bool)
		)
		ufops, err := dc.GetUfops(ctx)
		if err != nil {
			log.Fatalf("get ufops failed. %v", err)
		}
		for _, info := range ufops {
			detail, err := dc.GetUfopInfoDetail(ctx, info.Name)
			if err != nil {
				log.Fatalf("get ufop detail failed. %s %v", info.Name, err)
			}
			var releases = make([]appRelease, 0, len(detail.ReleaseDetail))
			for _, release := range detail.ReleaseDetail {
				var _release = appRelease{
					Version: release.Verstr,
					Image:   release.Image,
					Flavor:  release.Flavor,
					Ctime:   release.Ctime,
				}
				for _, instance := range release.Instances {
					_release.Expect += instance.Expect
					_release.Actual += instance.Actual
				}
				if _release.Expect > 0 || showALL {
					releases = append(releases, _release)
				}
			}
			apps[info.Name] = releases
		}
		// bs, _ := json.MarshalIndent(apps, "", "\t")
		// fmt.Println(string(bs))
		for name, rs := range apps {
			for _, r := range rs {
				fmt.Printf("%-20s\t%-15s\t%s\t%d|%d\n", name, r.Version, r.Image, r.Actual, r.Expect)
			}
		}
	}

}

func readJSON(fileName string, data interface{}) error {
	buf, err := ioutil.ReadFile(fileName)
	if err != nil {
		return errors.Wrapf(err, "read json %s", fileName)
	}
	err = json.Unmarshal(buf, data)
	if err != nil {
		return errors.Wrapf(err, "read json %s", fileName)
	}
	return nil
}

func readYAML(fileName string, data interface{}) error {
	buf, err := ioutil.ReadFile(fileName)
	if err != nil {
		return errors.Wrapf(err, "read json %s", fileName)
	}
	err = yaml.Unmarshal(buf, data)
	if err != nil {
		return errors.Wrapf(err, "read json %s", fileName)
	}
	return nil
}

func setUfopOfficial(ctx context.Context, app string, config doraController.Config) error {
	var (
		mac    = &qiniumac.Mac{AccessKey: config.AccessKey, SecretKey: []byte(config.SecretKey)}
		client = rpc.Client{Client: qiniumac.NewClient(mac, config.Transport)}
		url    = fmt.Sprintf("%s%s/ufops/%s/official", config.Host, "/v1", app)
	)
	return client.CallWithJson(ctx, nil, "POST", url,
		struct {
			Official bool `json:"official"`
		}{Official: true},
	)
}

func etcdGet(ctx context.Context, c *etcd.Client, key string, value interface{}) error {
	resp, err := etcd.NewKV(c).Get(ctx, key)
	if err != nil {
		return err
	}
	if resp.Kvs == nil || len(resp.Kvs) == 0 {
		return nil
	}
	return json.Unmarshal(resp.Kvs[0].Value, value)
}

func etcdPut(ctx context.Context, c *etcd.Client, key string, value interface{}) error {
	bs, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = etcd.NewKV(c).Put(ctx, key, string(bs))
	return err
}

func isNil(v interface{}) bool {
	if v == nil {
		return true
	}
	if v2, ok := v.(*interface{}); ok {
		return *v2 == nil
	}
	return false
}

func cleanupInterfaceArray(in []interface{}) []interface{} {
	res := make([]interface{}, len(in))
	for i, v := range in {
		res[i] = cleanupMapValue(v)
	}
	return res
}
func cleanupInterfaceMap(in map[interface{}]interface{}) map[string]interface{} {
	res := make(map[string]interface{})
	for k, v := range in {
		res[fmt.Sprintf("%v", k)] = cleanupMapValue(v)
	}
	return res
}
func cleanupMapValue(v interface{}) interface{} {
	if v == nil {
		return nil
	}
	switch v := v.(type) {
	case []interface{}:
		return cleanupInterfaceArray(v)
	case map[interface{}]interface{}:
		return cleanupInterfaceMap(v)
	case float32:
		return v
	case float64:
		return v
	case int:
		return v
	case int32:
		return v
	case int64:
		return v
	case string:
		return v
	default:
		return v
		// return fmt.Sprintf("%v", v)
	}
}
