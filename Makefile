build-k6:
	go clean -modcache
	mkdir -p /tmp/xk6 && \
	pushd /tmp/xk6 && \
	go install go.k6.io/xk6/cmd/xk6@latest && \
	xk6 build --with github.com/phymbert/xk6-sse@0abbe3e94fe104a13021524b1b98d26447a7d182 && \
	mkdir -p ~/.local/bin/ && \
	mv k6 ~/.local/bin/k6-sse && \
	popd