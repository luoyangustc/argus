package sand

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	"qiniu.com/argus/cap/model"
)

func TestSand(t *testing.T) {

	sm := NewSandMixer("^" + SandHeadStr)

	tasks := []model.TaskModel{
		model.TaskModel{
			URI: "url0",
		},
		model.TaskModel{
			URI: "url1",
		},
		model.TaskModel{
			URI: "url2",
			Labels: []model.LabelInfo{
				model.LabelInfo{
					Name: "pulp",
				},
			},
		},
	}
	err := sm.AddSand(context.Background(), tasks...)
	assert.NoError(t, err)

	task0 := model.TaskModel{
		TaskID: SandHeadStr + "0",
	}
	is := sm.IsSand(context.Background(), task0.TaskID)
	t.Log(is)

	err = sm.AddSandFileByURL("http://peiuccjvx.bkt.gdipper.com/bk_20.json")
	assert.NoError(t, err)

	// ret := sm.GetSandsByType(context.Background(), 10, "terror")
	// t.Log(ret)

	result := sm.Check(context.Background(), &model.TaskResult{
		TaskID: "QINIU_SAND_104603",
		URI:    "TEST",
		Labels: []model.LabelInfo{
			model.LabelInfo{
				Name: "pulp",
				Type: "classification",
			},
		},
	})
	t.Log(result)
}

func TestCompareDatas(t *testing.T) {
	//Test true
	sPulps := make([]interface{}, 0)
	sPulp := model.LabelData{
		Class: "pulp",
		Score: 0.89,
	}
	sPulps = append(sPulps, sPulp)

	dPulps := make([]interface{}, 0)
	dPulp := model.LabelData{
		Class: "pulp",
		Score: 0.89,
	}
	dPulps = append(dPulps, dPulp)

	resp := compareDatas("pulp", sPulps, dPulps)
	assert.Equal(t, true, resp)

	//test false
	sPos := make([]interface{}, 0)
	sPo := model.LabelPoliticianData{
		Class: "politician",
	}
	sPos = append(sPos, sPo)

	dPos := make([]interface{}, 0)
	dPo := model.LabelPoliticianData{
		Class: "normal",
	}
	dPos = append(dPos, dPo)

	resp = compareDatas("politician", sPos, dPos)
	assert.Equal(t, false, resp)
}
