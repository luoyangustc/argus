package httpserv

import (
	"errors"
	"net/http"

	"qiniupkg.com/x/log.v7"
)

type Config struct {
	LogPath        string `json:"log_path"`
	LogLevel       int    `json:"log_level"`
	BindHost       string `json:"bind_host"`
	StatusServHost string `json:"status_serv_host"`
	MgtServHost    string `json:"mgt_serv_host"`
}

type Service struct {
	Config
	//mutex     sync.Mutex
	//dataMutex sync.Mutex
}

func NewService(conf *Config) (s *Service, err error) {
	err = configCheck(conf)
	if err != nil {
		log.Error("start service fail ", err)
		return nil, err
	}

	s = new(Service)

	s.Config = *conf
	return s, err
}

func (s *Service) Run() (err error) {

	finish := make(chan error)

	go func() {
		var err error
		defer func() {
			finish <- err
		}()
		err = s.start()
	}()
	err = <-finish
	return err
}

func (s *Service) start() (err error) {

	mux := http.NewServeMux()
	registerHanders(s, mux)
	err = http.ListenAndServe(s.Config.BindHost, mux)
	/*
		router := gin.Default()
		router.GET("/v1/internal/device/:devid/sub_devices", func(c *gin.Context) {
			devid := c.Param("devid")
			c.String(http.StatusOK, "Hello %s", devid)
		})
		router.Run(":8080")
	*/
	return
}

func configCheck(conf *Config) (err error) {

	if conf == nil {
		err = errors.New("config file parse fail")
		return
	}

	if conf.BindHost == "" {
		err = errors.New("BindHost config nil")
		return
	}

	if conf.StatusServHost == "" {
		err = errors.New("StatusServerHost config nil")
		return
	}

	if conf.MgtServHost == "" {
		err = errors.New("MgtServHost config nil")
		return
	}

	return
}

func registerHanders(s *Service, mux *http.ServeMux) {

	mux.HandleFunc("/get_mgt_server", func(resp http.ResponseWriter, req *http.Request) {
		s.onGetMgtServer(resp, req)
	})

	mux.HandleFunc("/get_session_server", func(resp http.ResponseWriter, req *http.Request) {
		s.onGetSessionServer(resp, req)
	})

	mux.HandleFunc("/query_device_status", func(resp http.ResponseWriter, req *http.Request) {
		s.onQueryDeviceStatus(resp, req)
	})

	mux.HandleFunc("/live", func(resp http.ResponseWriter, req *http.Request) {
		s.onLive(resp, req)
	})

	mux.HandleFunc("/snap", func(resp http.ResponseWriter, req *http.Request) {
		s.onSnap(resp, req)
	})

	mux.HandleFunc("/device_mgr_update", func(resp http.ResponseWriter, req *http.Request) {
		s.onDeviceMgrUpdate(resp, req)
	})

	mux.HandleFunc("/v1/internal/", func(resp http.ResponseWriter, req *http.Request) {
		s.onInternalReq(resp, req)
	})

	mux.HandleFunc("/test/live", func(resp http.ResponseWriter, req *http.Request) {
		s.onTestLive(resp, req)
	})

	/*
		mux.HandleFunc("/v1/internal/sub_device", func(resp http.ResponseWriter, req *http.Request) {
			s.onQuerySubdeviceMgrInfo(resp, req)
		})

		mux.HandleFunc("/v1/internal/device/", func(resp http.ResponseWriter, req *http.Request) {
			s.onQueryDeviceMgrInfo(resp, req)
		})*/

	mux.HandleFunc("/", func(resp http.ResponseWriter, req *http.Request) {
		s.onLocation(resp, req)
	})

}
