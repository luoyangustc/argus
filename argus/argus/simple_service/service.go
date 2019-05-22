package simple_service

type Service interface {
	Version() string
	Config() interface{}

	Init(interface{}) error
}

//----------------------------------------------------------------------------//

type mockService struct {
}

func NewMockService() *mockService {
	return &mockService{}
}

func (s *mockService) Version() string              { return "v20171210" }
func (s *mockService) Config() interface{}          { return struct{}{} }
func (s *mockService) Init(_conf interface{}) error { return nil }
