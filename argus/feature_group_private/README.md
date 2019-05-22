`proto`: general proto definition
`image_feature`: image feature caculation
`manager`: storage tier
`search`: do search jobs

`group`: wrap image_feature, group, search as a service, which implements interface.go
`service`: wrap group as a web service , which  implements service.go
