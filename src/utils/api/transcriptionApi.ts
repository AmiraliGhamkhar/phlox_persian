import { handleApiRequest, universalFetch } from "../helpers/apiHelpers";
import { buildApiUrl, getApiBaseUrl, getRequestToken } from "../helpers/apiConfig";

export const transcriptionApi = {
    transcribeAudio: async (formData) => {
        return handleApiRequest({
            apiCall: async (signal) => {
                const url = await buildApiUrl(`/api/transcribe/audio`);
                return universalFetch(url, {
                    method: "POST",
                    body: formData,
                    signal: signal,
                });
            },
            errorMessage: "Error transcribing audio",
        });
    },

    reprocessTranscription: async (formData) => {
        return handleApiRequest({
            apiCall: async (signal) => {
                const url = await buildApiUrl(`/api/transcribe/reprocess`);
                return universalFetch(url, {
                    method: "POST",
                    body: formData,
                    signal: signal,
                });
            },
            timeout: 120000,
            errorMessage: "Error reprocessing transcription",
        });
    },

    transcribeDictation: async (formData) => {
        return handleApiRequest({
            apiCall: async () => {
                const url = await buildApiUrl(`/api/transcribe/dictate`);
                return universalFetch(url, {
                    method: "POST",
                    body: formData,
                });
            },
            errorMessage: "Error transcribing dictation",
        });
    },

    processDocument: async (formData) => {
        return handleApiRequest({
            apiCall: async (signal) => {
                const url = await buildApiUrl(
                    `/api/transcribe/process-document`,
                );
                return universalFetch(url, {
                    method: "POST",
                    body: formData,
                    signal,
                });
            },
            timeout: 180000,
            errorMessage: "Error processing document",
        });
    },

    extractDemographics: async (formData) => {
        return handleApiRequest({
            apiCall: async (signal) => {
                const url = await buildApiUrl(
                    `/api/transcribe/extract-demographics`,
                );
                return universalFetch(url, {
                    method: "POST",
                    body: formData,
                    signal,
                });
            },
            timeout: 180000,
            errorMessage: "Error extracting demographics from document",
        });
    },

    extractDemographicsFromText: async (payload) => {
        return handleApiRequest({
            apiCall: async (signal) => {
                const url = await buildApiUrl(
                    `/api/transcribe/extract-demographics-from-text`,
                );
                return universalFetch(url, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                    signal,
                });
            },
            timeout: 180000,
            errorMessage: "Error extracting demographics from text",
        });
    },

    extractDemographicsVisual: async (payload) => {
        return handleApiRequest({
            apiCall: async (signal) => {
                const url = await buildApiUrl(
                    `/api/transcribe/extract-demographics-visual`,
                );
                return universalFetch(url, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                    signal,
                });
            },
            timeout: 300000,
            errorMessage: "Error extracting demographics from visual document",
        });
    },

    processDocumentFromText: async (payload) => {
        return handleApiRequest({
            apiCall: async (signal) => {
                const url = await buildApiUrl(
                    `/api/transcribe/process-document-from-text`,
                );
                return universalFetch(url, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                    signal,
                });
            },
            timeout: 180000,
            errorMessage: "Error processing extracted document text",
        });
    },

    processDocumentVisual: async (payload) => {
        return handleApiRequest({
            apiCall: async (signal) => {
                const url = await buildApiUrl(
                    `/api/transcribe/process-document-visual`,
                );
                return universalFetch(url, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                    signal,
                });
            },
            timeout: 300000,
            errorMessage: "Error processing visual document",
        });
    },

    openLiveTranscription: async ({
        onPartial,
        onFinal,
        onError,
        onReady,
    }: {
        onPartial?: (text: string) => void;
        onFinal?: (text: string) => void;
        onError?: (message: string) => void;
        onReady?: (info: { authoritative: boolean }) => void;
    } = {}) => {
        const baseUrl = await getApiBaseUrl();
        const token = await getRequestToken();
        const wsBase = baseUrl
            ? baseUrl.replace(/^http/i, "ws")
            : `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}`;
        // Auth token travels in a WebSocket subprotocol, never in the URL:
        // query strings are written to uvicorn access logs and to the desktop
        // app log on disk (A09:2025).
        const subprotocols = token ? ["phlox-live", token] : ["phlox-live"];
        const socket = new WebSocket(`${wsBase}/api/transcribe/live`, subprotocols);

        let lastText = "";
        let authoritative = false;
        let opened = false;
        let settleStop: ((result: { text: string; authoritative: boolean }) => void) | null =
            null;

        const result = () => ({ text: lastText, authoritative });

        socket.binaryType = "arraybuffer";

        socket.onopen = () => {
            // Fail closed: an authenticated session must be confirmed by the
            // server via the negotiated subprotocol (token must not echo).
            if (token && socket.protocol !== "phlox-live") {
                onError?.("Live transcription authentication failed");
                try {
                    socket.close();
                } catch {
                    // already closed
                }
                return;
            }
            opened = true;
        };

        socket.onmessage = (event) => {
            if (typeof event.data !== "string") return;
            let payload: any;
            try {
                payload = JSON.parse(event.data);
            } catch {
                return;
            }
            if (payload?.type === "ready") {
                authoritative = Boolean(payload.authoritative);
                onReady?.({ authoritative });
                return;
            }
            if (payload?.type === "partial") {
                lastText = payload.text || lastText;
                onPartial?.(lastText);
                return;
            }
            if (payload?.type === "final") {
                lastText = payload.text || lastText;
                onFinal?.(lastText);
                if (settleStop) {
                    const done = settleStop;
                    settleStop = null;
                    done(result());
                }
                return;
            }
            if (payload?.type === "error") {
                onError?.(payload.message || "Live transcription error");
            }
        };

        const sendPcm = (samples: Int16Array) => {
            if (socket.readyState !== WebSocket.OPEN || !samples?.length) return;
            // Speechmatics documents that sending audio faster than the engine
            // reads it can fill TCP buffers and close the socket "with
            // prejudice". Skip frames while the browser is already backed up
            // (~1 MB) rather than overrun the connection.
            if (socket.bufferedAmount > 1_000_000) return;
            // Zero-copy view over the exact byte range. The explicit
            // ArrayBuffer generic satisfies WebSocket.send() under TS 6
            // (ArrayBufferLike is no longer assignable to BufferSource).
            const bytes = new Uint8Array<ArrayBuffer>(
                samples.buffer as ArrayBuffer,
                samples.byteOffset,
                samples.byteLength,
            );
            socket.send(bytes);
        };

        const close = () => {
            try {
                if (
                    socket.readyState === WebSocket.OPEN ||
                    socket.readyState === WebSocket.CONNECTING
                ) {
                    socket.close();
                }
            } catch {
                // already closed
            }
        };

        const stop = () =>
            new Promise<{ text: string; authoritative: boolean }>((resolve) => {
                if (socket.readyState !== WebSocket.OPEN) {
                    resolve(result());
                    return;
                }
                const timer = window.setTimeout(() => {
                    settleStop = null;
                    resolve(result());
                    close();
                }, 8000);
                settleStop = (value) => {
                    window.clearTimeout(timer);
                    resolve(value);
                };
                try {
                    socket.send(JSON.stringify({ type: "stop" }));
                } catch {
                    window.clearTimeout(timer);
                    resolve(result());
                }
            });

        await new Promise<void>((resolve, reject) => {
            const timer = window.setTimeout(() => {
                if (!opened) {
                    close();
                    reject(new Error("Live transcription timed out"));
                }
            }, 8000);
            socket.onopen = () => {
                opened = true;
                window.clearTimeout(timer);
                resolve();
            };
            socket.onerror = () => {
                if (!opened) {
                    window.clearTimeout(timer);
                    reject(new Error("Live transcription unavailable"));
                } else {
                    onError?.("Live transcription connection lost");
                }
            };
            socket.onclose = () => {
                if (settleStop) {
                    const done = settleStop;
                    settleStop = null;
                    done(result());
                }
            };
        });

        return { sendPcm, stop, close };
    },
};
