# syntax=docker/dockerfile:1

# p3 KV-cache / prefix-cache demos. The runtime stage is Chainguard's
# distroless static image: no shell. policy.txt is compiled in via go:embed.
#
#   docker build -t llm-demos-p3 ./p3
#   docker run --rm llm-demos-p3
#
# prefix_cache reaches the laptop's single-slot llama-server via
# host.docker.internal. BASE_URL is the native API root, not /v1:
#
#   docker run --rm --entrypoint /usr/bin/prefix_cache \
#     -e BASE_URL=http://host.docker.internal:8393 \
#     llm-demos-p3
FROM cgr.dev/chainguard/go:latest-dev AS build

USER root
WORKDIR /src

COPY go.mod ./
RUN go mod download

COPY . .
ENV CGO_ENABLED=0
RUN go build -o /out/kv_math ./cmd/kv_math \
    && go build -o /out/prefix_cache ./cmd/prefix_cache

FROM cgr.dev/chainguard/static:latest

COPY --from=build /out/ /usr/bin/
ENTRYPOINT ["/usr/bin/kv_math"]
