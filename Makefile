build-k6:
	mkdir -p /tmp/xk6 && \
	pushd /tmp/xk6 && \
	go install go.k6.io/xk6/cmd/xk6@latest && \
	GOFLAGS="-mod=mod" xk6 build master --with github.com/andrewrreed/xk6-sse@a24fd84 && \
	mkdir -p ~/.local/bin/ && \
	mv k6 ~/.local/bin/k6-sse && \
	popd