# syntax=docker/dockerfile:1

# latest-dev has the Go toolchain and a shell, and is only used to compile.
# The runtime stage is Chainguard's distroless static image: no shell.
FROM cgr.dev/chainguard/go:latest-dev AS build

USER root
WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .
ENV CGO_ENABLED=0
RUN go build -o /out/tokenize_demo ./cmd/tokenize_demo \
    && go build -o /out/stream_ttft ./cmd/stream_ttft \
    && go build -o /out/kv_cache_math ./cmd/kv_cache_math

FROM cgr.dev/chainguard/static:latest

COPY --from=build /out/ /usr/bin/
ENTRYPOINT ["/usr/bin/tokenize_demo"]
