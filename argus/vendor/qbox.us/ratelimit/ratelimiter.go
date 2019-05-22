package ratelimit

import (
	"time"

	"gopkg.in/bsm/ratelimit.v1"
)

type RateLimiter struct {
	rate int
	per  time.Duration
	*ratelimit.RateLimiter
}

func New(rate int, per time.Duration) *RateLimiter {
	return &RateLimiter{
		rate:        rate,
		per:         per,
		RateLimiter: ratelimit.New(rate, per),
	}
}

func (rl *RateLimiter) Limit() bool {
	if rl.rate < 1 {
		return true
	}
	return rl.RateLimiter.Limit()
}
