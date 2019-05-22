package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"

	"github.com/k0kubun/pp"
	"qiniu.com/argus/feature_group_private/proto"
	"qiniu.com/argus/feature_group_private/search"
	cpu "qiniu.com/argus/feature_group_private/search/cpu"
	gpu "qiniu.com/argus/feature_group_private/search/gpu"
)

const (
	BLOCK_SIZE = 32 * 1024 * 1024 // 32M
	BATCH_SIZE = 100
)

func createSet(ctx context.Context, mode string, dimension, precision, num int) (set search.Set, err error) {
	blockFeatureCount := BLOCK_SIZE / (dimension * precision)
	blockNum := (num + blockFeatureCount - 1) / blockFeatureCount
	setName := search.SetName("test")

	config := search.Config{
		Precision: precision,
		Dimension: dimension,
		BlockSize: BLOCK_SIZE,
		BlockNum:  blockNum,
		BatchSize: BATCH_SIZE,
		Version:   0,
	}

	pp.Printf("block size: %v , num: %v \n", blockFeatureCount, blockNum)

	var sets search.Sets

	if mode == "gpu" {
		sets, err = gpu.NewSets(config)
	} else {
		sets, err = cpu.NewSets(config)
	}

	if err != nil {
		return
	}

	err = sets.New(ctx, setName, config, cpu.SetStateCreated)
	if err != nil {
		return
	}

	set, err = sets.Get(ctx, setName)
	if err != nil {
		return
	}

	features, _ := createFeatures(dimension, precision, blockFeatureCount)
	for i := 0; i < num/blockFeatureCount; i++ {
		err = set.Add(ctx, features...)
		if err != nil {
			return
		}
	}
	return
}

func createFeatures(dim, pre, num int) (features []proto.Feature, values []proto.FeatureValue) {
	for i := 0; i < num; i++ {
		feature := proto.Feature{
			ID: proto.FeatureID(search.GetRandomString(12)),
		}
		r := rand.New(rand.NewSource(time.Now().Unix()))
		for j := 0; j < dim; j++ {
			bs := make([]byte, pre)
			binary.LittleEndian.PutUint32(bs, math.Float32bits(r.Float32()*2-1))
			feature.Value = append(feature.Value, proto.FeatureValue(bs)...)
		}
		features = append(features, feature)
		values = append(values, feature.Value)
	}
	return
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	ctx := context.Background()
	total := flag.Float64("total", 1.0, "total num(1m) of features")
	num := flag.Int("num", 1, "search num")
	dimension := flag.Int("dimension", 512, "dimension of feature, 512 or 4096")
	times := flag.Int("times", 50, "search times")
	verbose := flag.Bool("verbose", false, "verbose")
	mode := flag.String("mode", "cpu", "mode cpu or gpu")
	flag.Parse()

	precision := 4
	*total = math.Floor(*total * 1000 * 1000)

	set, err := createSet(ctx, *mode, *dimension, precision, int(*total))
	if err != nil {
		return
	}

	pp.Printf("searching %v in %v \n", *num, int(*total))
	_, values := createFeatures(*dimension, precision, *num)

	Start := time.Now()
	for i := 0; i < *times; i++ {
		start := time.Now()
		_, _ = set.Search(ctx, -1.0, 1, values...)
		if *verbose {
			fmt.Printf("%v, %.2f\n", i, float64((time.Now().UnixNano()-start.UnixNano()))/1e9)
		}
	}
	totalTimeCost := float64((time.Now().UnixNano() - Start.UnixNano())) / 1e9
	fmt.Printf("total: %.6f, average %.6f\n", totalTimeCost, totalTimeCost/(float64(*times)))
}
