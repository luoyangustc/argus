package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"
	"time"

	etcd "github.com/coreos/etcd/clientv3"

	"github.com/qiniu/log.v1"
)

func etcdMain(args []string) {

	var (
		flagSet = flag.NewFlagSet("etcd", flag.ExitOnError)

		endpoints string
	)
	flagSet.StringVar(&endpoints, "endpoints", "",
		"end points, like: http://10.200.30.13:2379,http://10.200.30.14:2379")

	flagSet.Parse(args)
	args = flagSet.Args()

	var (
		config = etcd.Config{
			Endpoints:   strings.Split(endpoints, ","),
			DialTimeout: time.Second,
		}
		client, err = etcd.New(config)
	)
	if err != nil {
		log.Fatalf("new etcd client failed. %#v %v", config, err)
	}
	var kv = etcd.NewKV(client)
	var key = args[1]

	switch args[0] {
	case "list":
		resp, err := kv.Get(context.Background(), key, etcd.WithPrefix())
		if err != nil {
			log.Fatalf("get prefix failed. %s %v", key, err)
		}
		var rest = make([]string, 0)
		for _, kv := range resp.Kvs {
			rest = append(rest, string(kv.Key))
		}
		sort.Strings(rest)
		for _, str := range rest {
			fmt.Println(str)
		}
	case "show":
		resp, err := kv.Get(context.Background(), key)
		if err != nil {
			log.Fatalf("get key failed. %s %v", key, err)
		}
		if len(resp.Kvs) == 0 {
			log.Fatalf("no key found. %s", key)
		}
		var v interface{}
		if err = json.Unmarshal(resp.Kvs[0].Value, &v); err != nil {
			log.Fatalf("unmarshal value failed. %s %v", string(resp.Kvs[0].Value), err)
		}
		var encode = json.NewEncoder(os.Stdout)
		encode.SetIndent("", "\t")
		if err = encode.Encode(v); err != nil {
			log.Fatalf("encode value failed. %#v %v", v, err)
		}
	case "check":
		var (
			kvs1 = []etcdKV{}
			kvs2 = []etcdKV{}
		)
		if err := json.NewDecoder(os.Stdin).Decode(&kvs1); err != nil {
			log.Fatalf("decode config failed. %v", err)
		}
		sort.Sort(byEtcdKey(kvs1))
		resp, err := kv.Get(context.Background(), key, etcd.WithPrefix())
		if err != nil {
			log.Fatalf("get prefix failed. %s %v", key, err)
		}
		for _, kv := range resp.Kvs {
			var v interface{}
			json.Unmarshal(kv.Value, &v)
			kvs2 = append(kvs2, etcdKV{Key: strings.TrimPrefix(string(kv.Key), key), Value: v})
		}
		sort.Sort(byEtcdKey(kvs2))
		var i, j, n, m = 0, 0, len(kvs1), len(kvs2)
		for i < n && j < m {
			if kvs1[i].Key < kvs2[j].Key {
				fmt.Printf(">> %s\n", kvs1[i].Key)
				i++
			} else if kvs1[i].Key > kvs2[j].Key {
				fmt.Printf("<< %s\n", kvs2[j].Key)
				j++
			} else {
				// var v1, v2 interface{}
				// json.Unmarshal([]byte(kvs1[i].Value), &v1)
				// json.Unmarshal([]byte(kvs2[j].Value), &v2)
				if !reflect.DeepEqual(kvs1[i].Value, kvs2[j].Value) {
					fmt.Printf("|| %s\n", kvs1[i].Key)
					{
						fmt.Print("\t>> ")
						encoder := json.NewEncoder(os.Stdout)
						encoder.SetIndent("\t>> ", "\t")
						encoder.Encode(kvs1[i].Value)
					}
					{
						fmt.Print("\t<< ")
						encoder := json.NewEncoder(os.Stdout)
						encoder.SetIndent("\t<< ", "\t")
						encoder.Encode(kvs2[j].Value)
					}
				}
				i++
				j++
			}
		}
		for ; i < n; i++ {
			fmt.Printf(">> %s\n", kvs1[i].Key)
		}
		for ; j < m; j++ {
			fmt.Printf("<< %s\n", kvs2[j].Key)
		}
	case "init":
		var (
			kvs1 = []etcdKV{}
			kvs2 = []etcdKV{}
		)
		if err := json.NewDecoder(os.Stdin).Decode(&kvs1); err != nil {
			log.Fatalf("decode config failed. %v", err)
		}
		sort.Sort(byEtcdKey(kvs1))
		resp, err := kv.Get(context.Background(), key, etcd.WithPrefix())
		if err != nil {
			log.Fatalf("get prefix failed. %s %v", key, err)
		}
		for _, kv := range resp.Kvs {
			var v interface{}
			json.Unmarshal(kv.Value, &v)
			kvs2 = append(kvs2, etcdKV{Key: strings.TrimPrefix(string(kv.Key), key), Value: v})
		}
		sort.Sort(byEtcdKey(kvs2))
		var i, j, n, m = 0, 0, len(kvs1), len(kvs2)
		for i < n && j < m {
			if kvs1[i].Key < kvs2[j].Key {
				bs, _ := json.Marshal(kvs1[i].Value)
				kv.Put(context.Background(), key+kvs1[i].Key, string(bs))
				i++
			} else if kvs1[i].Key > kvs2[j].Key {
				kv.Delete(context.Background(), key+kvs2[j].Key)
				j++
			} else {
				// var v1, v2 interface{}
				// json.Unmarshal([]byte(kvs1[i].Value), &v1)
				// json.Unmarshal([]byte(kvs2[j].Value), &v2)
				if !reflect.DeepEqual(kvs1[i].Value, kvs2[j].Value) {
					bs, _ := json.Marshal(kvs1[i].Value)
					kv.Put(context.Background(), key+kvs1[i].Key, string(bs))
				}
				i++
				j++
			}
		}
		for ; i < n; i++ {
			bs, _ := json.Marshal(kvs1[i].Value)
			kv.Put(context.Background(), key+kvs1[i].Key, string(bs))
		}
		for ; j < m; j++ {
			kv.Delete(context.Background(), key+kvs2[j].Key)
		}
	case "update":
		var (
			kvs1 = []etcdKV{}
			kvs2 = []etcdKV{}
		)
		if err := json.NewDecoder(os.Stdin).Decode(&kvs1); err != nil {
			log.Fatalf("decode config failed. %v", err)
		}
		sort.Sort(byEtcdKey(kvs1))
		resp, err := kv.Get(context.Background(), key, etcd.WithPrefix())
		if err != nil {
			log.Fatalf("get prefix failed. %s %v", key, err)
		}
		for _, kv := range resp.Kvs {
			var v interface{}
			json.Unmarshal(kv.Value, &v)
			kvs2 = append(kvs2, etcdKV{Key: strings.TrimPrefix(string(kv.Key), key), Value: v})
		}
		sort.Sort(byEtcdKey(kvs2))
		var i, j, n, m = 0, 0, len(kvs1), len(kvs2)
		for i < n && j < m {
			if kvs1[i].Key < kvs2[j].Key {
				bs, _ := json.Marshal(kvs1[i].Value)
				kv.Put(context.Background(), key+kvs1[i].Key, string(bs))
				i++
			} else if kvs1[i].Key > kvs2[j].Key {
				// kv.Delete(context.Background(), key+kvs2[j].Key)
				j++
			} else {
				// var v1, v2 interface{}
				// json.Unmarshal([]byte(kvs1[i].Value), &v1)
				// json.Unmarshal([]byte(kvs2[j].Value), &v2)
				if !reflect.DeepEqual(kvs1[i].Value, kvs2[j].Value) {
					bs, _ := json.Marshal(kvs1[i].Value)
					kv.Put(context.Background(), key+kvs1[i].Key, string(bs))
				}
				i++
				j++
			}
		}
		for ; i < n; i++ {
			bs, _ := json.Marshal(kvs1[i].Value)
			kv.Put(context.Background(), key+kvs1[i].Key, string(bs))
		}
		for ; j < m; j++ {
			// kv.Delete(context.Background(), key+kvs2[j].Key)
		}
	case "dump":
		var rest = []etcdKV{}
		resp, err := kv.Get(context.Background(), key, etcd.WithPrefix())
		if err != nil {
			log.Fatalf("get prefix failed. %s %v", key, err)
		}
		for _, kv := range resp.Kvs {
			rest = append(rest, etcdKV{Key: strings.TrimPrefix(string(kv.Key), key), Value: string(kv.Value)})
		}
		sort.Sort(byEtcdKey(rest))
		var encode = json.NewEncoder(os.Stdout)
		encode.SetIndent("", "\t")
		if err = encode.Encode(rest); err != nil {
			log.Fatalf("encode value failed. %#v %v", rest, err)
		}
	}
}

type etcdKV struct {
	Key   string
	Value interface{}
}
type byEtcdKey []etcdKV

func (a byEtcdKey) Len() int           { return len(a) }
func (a byEtcdKey) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byEtcdKey) Less(i, j int) bool { return a[i].Key < a[j].Key }
