# Rakuvo gateway image.
#
#   docker build -t rakuvo --build-arg VERSION=0.1.0 .
#   docker run --rm -p 8080:8080 -e OPENROUTER_API_KEY -e RAKUVO_TOKEN rakuvo serve -provider openrouter \
#     -listen 0.0.0.0:8080 -token-env RAKUVO_TOKEN -guard -tool-policy-auto -loop-guard
#
# The image holds only the compiled program: no source, no shell, no package manager, and it runs as a
# non-root user. Pass keys as environment variables at run time; never bake them into an image.

ARG GO_VERSION=1.27
FROM golang:${GO_VERSION}-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ARG VERSION=dev
ENV CGO_ENABLED=0
RUN go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" -o /out/rakuvo ./cmd/rakuvo

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/rakuvo /rakuvo
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD ["/rakuvo", "healthcheck", "-url", "http://127.0.0.1:8080/healthz"]
ENTRYPOINT ["/rakuvo"]
CMD ["serve", "-listen", "0.0.0.0:8080", "-token-env", "RAKUVO_TOKEN"]
