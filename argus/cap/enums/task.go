package enums

//=========================================================================//
type TaskStatusType string

const (
	TaskTodo  string = "todo"
	TaskDoing string = "doing"
	TaskDone  string = "done"
)

//=======================================================================//
type TaskLabelType string

const (
	LableClassification TaskLabelType = "classification"
	LabelDetection      TaskLabelType = "detection"
)

func (this TaskLabelType) IsValid() bool {
	switch this {
	case LableClassification, LabelDetection:
		return true
	default:
		return false
	}
}

//========================================================================//
type TaskType string

const (
	TaskPulp       = "pulp"
	TaskTerror     = "terror"
	TaskPolitician = "politician"
)

func (this TaskType) IsValid() bool {
	switch this {
	case TaskPulp, TaskTerror, TaskPolitician:
		return true
	default:
		return false
	}
}
