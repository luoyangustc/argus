#ifndef __TO_STRING_UTIL_H__
#define __TO_STRING_UTIL_H__

	template<typename t> static std::string calc_flux_unit(t flux)
	{
		char buf[100] = {0};
		string ret;
		if(flux<1024)
		{
			sprintf(buf,"%dB",(int)flux);
		}
		else if(flux < (1024*1024))
		{
			sprintf(buf,"%0.2fKB",(double)flux/1024);
		}
		else if(flux < (1024*1024*1024))
		{
			sprintf(buf,"%0.2fMB",(double)flux/(1024*1024));
		}
		else
			sprintf(buf,"%0.2fGB",(double)flux/(1024*1024*1024));
	
		ret = buf;
	
		return ret;
	};

	template<typename T> static std::string calc_speed_unit(T speed)
	{
		char buf[100] = {0};
		string ret;
		if(speed<1024)
			sprintf(buf,"%dbps",(int)speed);
		else if(speed<(1024*1024))
			sprintf(buf,"%0.1fKbps",(double)speed/1024);
		else
			sprintf(buf,"%0.1fMbps",(double)speed/(1024*1024));

		ret = buf;

		return ret;
	}


	template<typename T> static std::string calc_time_unit(T time)
	{
		char buf[100] = {0};
		string ret;
		if(time<60)
		{
			sprintf(buf,"%ds",(int)time);
		}
		else if(time<(60*60))
		{
			sprintf(buf,"%dm%ds",(int)time/60,(int)time%60);
		}
		else if( time< (60*60*24) )
		{
			int h = time/(60*60);
			int m = (time-h*60*60)/60;
			int s = time%60;
			sprintf(buf,"%dh%dm%ds",h,m,s);
		}
		else
		{
			int d = time/(60*60*24);
			int h = (time-d*60*60*24)/(60*60);
			int m = (time-d*60*60*24-h*60*60)/60;
			int s = time%60;
			sprintf(buf,"%dd%dh%dm%ds",d,h,m,s);
		}

		ret = buf;

		return ret;
	}


	template<typename T> static std::string calc_number_unit(T number)
	{
		char buf[100] = {0};
		string ret;
		if(number<1000)
		{
			sprintf(buf,"%d",(int)number);
		}
		else if(number<(1000*1000))
		{
			sprintf(buf,"%.1fK",number/1000.0);
		}
		else 
		{
			sprintf(buf,"%.2fM",number/1000.0/1000.0);
		}

		ret = buf;

		return ret;
	}

#endif //__TO_STRING_UTIL_H__
