# syntax=docker/dockerfile:1

# p2 support-triage demos. latest-dev has the Go toolchain and a shell, and
# is only used to compile. The runtime stage is Chainguard's distroless
# static image: no shell.
#
#   docker build -t llm-demos-p2 ./p2
#   docker run --rm llm-demos-p2
#
# Server-backed demos reach the laptop's llama-server via
# host.docker.internal:
#
#   docker run --rm --entrypoint /usr/bin/stream_meter \
#     -e BASE_URL=http://host.docker.internal:8080/v1 \
#     llm-demos-p2
FROM cgr.dev/chainguard/go:latest-dev AS build

USER root
WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .
ENV CGO_ENABLED=0
RUN go build -o /out/tokens_demo ./cmd/tokens_demo \
    && go build -o /out/context_math ./cmd/context_math \
    && go build -o /out/cost_calc ./cmd/cost_calc \
    && go build -o /out/stream_meter ./cmd/stream_meter \
    && go build -o /out/temperature_demo ./cmd/temperature_demo \
    && go build -o /out/structured_output ./cmd/structured_output

FROM cgr.dev/chainguard/static:latest

COPY --from=build /out/ /usr/bin/
ENTRYPOINT ["/usr/bin/tokens_demo"]
