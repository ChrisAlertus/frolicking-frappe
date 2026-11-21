# syntax=docker/dockerfile:1

# p4 vector-search demos. The runtime stage is Chainguard's distroless
# static image, so the build is CGO_ENABLED=0. HNSW is pure Go
# (github.com/coder/hnsw). libfaiss is not linked.
#
#   docker build -t llm-demos-p4 ./p4
#   docker run --rm llm-demos-p4 -n 20000
#
# The default -n is 1,000,000 (about 3.1 GB) and is OOM-killed by Docker
# Desktop's default ~2 GB VM (exit 137). Pass -n, or raise the VM limit.
#
# embed_rate, hybrid, and embed_words need the laptop's embedding server
# (./p4/setup.sh), reached via host.docker.internal:
#
#   docker run --rm --entrypoint /usr/bin/hybrid \
#     -e EMBED_URL=http://host.docker.internal:8395/v1 \
#     llm-demos-p4
FROM cgr.dev/chainguard/go:latest-dev AS build

USER root
WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .
ENV CGO_ENABLED=0
RUN go build -o /out/embed_rate ./cmd/embed_rate \
    && go build -o /out/hybrid ./cmd/hybrid \
    && go build -o /out/embed_words ./cmd/embed_words \
    && go build -o /out/real_bench ./cmd/real_bench \
    && go build -o /out/ann_bench ./cmd/ann_bench \
    && go build -o /out/fresh_filter ./cmd/fresh_filter

FROM cgr.dev/chainguard/static:latest

COPY --from=build /out/ /usr/bin/
ENTRYPOINT ["/usr/bin/ann_bench"]
