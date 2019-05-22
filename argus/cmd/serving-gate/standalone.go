package main

import (
	"context"
	"encoding/csv"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/qiniu/xlog.v1"

	cconf "qbox.us/cc/config"
	jsonlog "qbox.us/http/audit/jsonlog.v3"

	"qiniu.com/argus/atserving/model"
	"qiniu.com/argus/serving_gate"
)

// Standalone config
type StandaloneConfig struct {
	HTTPPort   int            `json:"http_port"`
	AuditLog   jsonlog.Config `json:"audit_log"`
	DebugLevel int            `json:"debug_level"`
	TimeOut    time.Duration  `json:"timeout"` // TODO

	InstanceFile string `json:"instance_file"`
}

func main4Standalone(ctx context.Context) {
	var (
		xl = xlog.FromContextSafe(ctx)

		conf = StandaloneConfig{}
		err  error

		logPush *gate.LogPushClient
		evals   gate.Evals
		_gate   gate.Gate
	)

	if err = cconf.Load(&conf); err != nil {
		xl.Fatalf("Failed to load configure file! %v", err)
	}
	xl.Infof("load conf %#v", conf)

	evalClients := gate.NewEvalClients()
	{
		hosts, err := collectCmdHosts(ctx, conf.InstanceFile)
		if err != nil {
			xl.Fatalf("init config failed. %v", err)
		}
		evalClients.SetHosts(hosts, conf.TimeOut)
	}

	evals = gate.NewEvals()
	evals.SetAppMetadataDefault(model.ConfigAppMetadata{
		Public: true,
	})

	logPush = gate.NewLogPushClient(gate.LogPushConfig{Open: false}, http.DefaultClient, nil)

	_gate = gate.NewGateStandalone(evals, evalClients)

	server(
		strconv.Itoa(conf.HTTPPort),
		&conf.AuditLog,
		conf,
		logPush, evals, _gate,
		nil,
	)
}

func collectCmdHosts(ctx context.Context, file string) (map[string]map[string][]string, error) {

	var (
		xl    = xlog.FromContextSafe(ctx)
		hosts = make(map[string]map[string][]string)
	)

	f, err := os.Open(file)
	if err != nil {
		xl.Errorf("collectCmdHosts open instance file error:%v", err)
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	for _, record := range records[1:] {
		if len(record) < 3 {
			continue
		}
		if _, ok := hosts[record[0]]; !ok {
			hosts[record[0]] = make(map[string][]string)
		}
		hosts[record[0]][record[1]] = append(hosts[record[0]][record[1]], record[2])

	}
	xl.Infof("cmd host list:%v", hosts)
	return hosts, nil
}
