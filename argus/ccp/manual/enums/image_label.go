package enums

const (
	PULP_normal = "normal" //-> PASS
	PULP_sexy   = "sexy"   //-> PASS
	PULP_pulp   = "pulp"   //-> BLOCK
)

const (
	TERROR_normal       = "normal"       //-> PASS
	TERROR_bloodiness   = "bloodiness"   //-> REVIEW
	TERROR_bomb         = "bomb"         //-> REVIEW
	TERROR_beheaded     = "beheaded"     //-> BLOCK
	TERROR_march_banner = "march_banner" //-> REVIEW
	TERROR_march_crowed = "march_crowed" //-> REVIEW
	TERROR_fight_police = "fight_police" //-> REVIEW
	TERROR_fight_person = "fight_person" //-> REVIEW
	TERROR_illegal_flag = "illegal_flag" //-> BLOCK
	TERROR_knives       = "knives"       //-> REVIEW
	TERROR_guns         = "guns"         //-> BLOCK
)

const (
	POLITICIAN_normal               = "normal"               //-> PASS
	POLITICIAN_affairs_official_gov = "affairs_official_gov" //-> BLOCK
	POLITICIAN_affairs_official_ent = "affairs_official_ent" //-> BLOCK
	POLITICIAN_anti_china_people    = "anti_china_people"    //-> REVIEW
	POLITICIAN_terrorist            = "terrorist"            //-> REVIEW
	POLITICIAN_affairs_celebrity    = "affairs_celebrity"    //-> REVIEW
	POLITICIAN_domestic_statesman   = "domestic_statesman"   //-> REVIEW
	POLITICIAN_foreign_statesman    = "foreign_statesman"    //-> REVIEW
)
