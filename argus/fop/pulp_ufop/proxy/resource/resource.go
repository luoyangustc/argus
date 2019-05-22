package proxy_resource

type Resource interface {
	// upload the file to the resource server
	// return the address of the resource
	Upload(filepath, key string) (string, error)
	// delete the resource from the server
	Delete(url string) error
}
