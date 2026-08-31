import { useEffect, useState, useRef, useCallback } from "react";
import { useTranscription } from "../../utils/hooks/useTranscription";
import { settingsService } from "../../utils/settings/settingsUtils";
import { settingsApi } from "../../utils/api/settingsApi";
import { transcriptionApi } from "../../utils/api/transcriptionApi";
import { AudioRecorder } from "../../utils/audioRecorder";

// Hook to manage scribe state and logic
// This can be used by ScribePillBox to control recording
export const useScribe = ({
    name,
    dob,
    gender,
    template,
    noteId,
    handleTranscriptionComplete,
    setLoading,
    onSendStart,
}) => {
    const [isAmbient, setIsAmbient] = useState(true);
    const [requireConsent, setRequireConsent] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [timer, setTimer] = useState(0);
    const audioRecorderRef = useRef(null);
    const timerIntervalRef = useRef(null);

    const [sendError, setSendError] = useState(null); // null = ok, else { message }
    const [liveTranscript, setLiveTranscript] = useState("");
    const [liveError, setLiveError] = useState(null);
    const lastFailedRef = useRef({ blob: null, meta: null, isAmbient: null });
    const liveSessionRef = useRef(null);
    // lastFailedRef shape: { blob, meta, isAmbient, liveResult }

    // Transcription API
    const { transcribeAudio, reprocessTranscription, isTranscribing } = useTranscription((data) => {
        if (data?.error) return;
        handleTranscriptionComplete({
            fields: data.fields,
            rawTranscription: data.rawTranscription,
            transcriptionDuration: data.transcriptionDuration,
            processDuration: data.processDuration,
        });
    }, setLoading);

    // Fetch user settings on mount
    useEffect(() => {
        const fetchSettings = async () => {
            try {
                const data = await settingsApi.fetchUserSettings();
                if (data && typeof data.scribe_is_ambient === "boolean") {
                    setIsAmbient(data.scribe_is_ambient);
                }
                setRequireConsent(
                    Boolean(data?.advanced_options?.require_scribe_consent),
                );
            } catch (error) {
                console.error("Error fetching user settings:", error);
            }
        };
        fetchSettings();
    }, []);

    // Timer effect
    useEffect(() => {
        if (isRecording && !isPaused) {
            timerIntervalRef.current = setInterval(() => {
                setTimer((prev) => prev + 1);
            }, 1000);
        } else {
            clearInterval(timerIntervalRef.current);
        }
        return () => clearInterval(timerIntervalRef.current);
    }, [isRecording, isPaused]);

    const closeLiveSession = useCallback(() => {
        if (liveSessionRef.current) {
            try {
                liveSessionRef.current.close();
            } catch {
                // already closed
            }
            liveSessionRef.current = null;
        }
    }, []);

    const resetRecordingState = useCallback(() => {
        setIsRecording(false);
        setIsPaused(false);
        setTimer(0);
        setSendError(null);
        setLiveTranscript("");
        setLiveError(null);
        lastFailedRef.current = { blob: null, meta: null, isAmbient: null };
        closeLiveSession();
        if (audioRecorderRef.current) {
            audioRecorderRef.current.stop().catch(() => {});
        }
        audioRecorderRef.current = null;
    }, [closeLiveSession]);

    // Reset when patient changes
    useEffect(() => {
        resetRecordingState();
    }, [name, dob, gender, resetRecordingState]);

    const clearLastFailed = useCallback(() => {
        lastFailedRef.current = { blob: null, meta: null, isAmbient: null };
        setSendError(null);
    }, []);

    const sendForTranscription = useCallback(
        async (blob, meta, liveResult = null) => {
            try {
                if (liveResult?.authoritative && liveResult?.text) {
                    await reprocessTranscription(
                        liveResult.text,
                        meta,
                        0,
                        isAmbient,
                    );
                } else {
                    await transcribeAudio(blob, meta, isAmbient);
                }
                clearLastFailed();
                return true;
            } catch (error) {
                console.error("Transcription failed:", error);
                lastFailedRef.current = {
                    blob,
                    meta: { ...meta },
                    isAmbient,
                    liveResult,
                };
                const message = error?.message || "پیاده‌سازی ناموفق بود";
                setSendError({ message });
                return false;
            }
        },
        [transcribeAudio, reprocessTranscription, isAmbient, clearLastFailed],
    );

    const retrySend = useCallback(async () => {
        const { blob, meta, isAmbient: ambient, liveResult } = lastFailedRef.current;
        if (!blob && !(liveResult?.authoritative && liveResult?.text)) return false;
        try {
            if (liveResult?.authoritative && liveResult?.text) {
                await reprocessTranscription(liveResult.text, meta, 0, ambient);
            } else {
                await transcribeAudio(blob, meta, ambient);
            }
            clearLastFailed();
            return true;
        } catch (error) {
            console.error("Transcription retry failed:", error);
            const message = error?.message || "پیاده‌سازی ناموفق بود";
            setSendError({ message });
            return false;
        }
    }, [transcribeAudio, reprocessTranscription, clearLastFailed]);

    const downloadLastRecording = useCallback(() => {
        const { blob } = lastFailedRef.current;
        if (!blob) return;
        const ext = blob.type.includes("wav")
            ? "wav"
            : blob.type.includes("webm")
              ? "webm"
              : "audio";
        const stamp = new Date()
            .toISOString()
            .replace(/[:.]/g, "-")
            .slice(0, 19);
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `recording_${stamp}.${ext}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, []);

    const dismissSendError = useCallback(() => {
        clearLastFailed();
    }, [clearLastFailed]);

    const startRecording = useCallback(async () => {
        try {
            const recorder = new AudioRecorder();
            setLiveTranscript("");
            setLiveError(null);
            try {
                const session = await transcriptionApi.openLiveTranscription({
                    onPartial: (text) => setLiveTranscript(text || ""),
                    onFinal: (text) => setLiveTranscript(text || ""),
                    onError: (message) => {
                        console.debug("Live transcription:", message);
                        setLiveError(message);
                    },
                });
                liveSessionRef.current = session;
                recorder.onPcm = (samples) => session.sendPcm(samples);
            } catch (error) {
                console.debug("Live transcription unavailable:", error);
                setLiveError(error?.message || String(error));
            }
            await recorder.start();
            audioRecorderRef.current = recorder;
            setIsRecording(true);
            setTimer(0);
        } catch (error) {
            console.error("Error starting recording:", error);
            closeLiveSession();
            alert("دسترسی به میکروفون ممکن نبود. لطفاً مجوزها را بررسی کنید.");
        }
    }, [closeLiveSession]);

    const pauseRecording = useCallback(() => {
        audioRecorderRef.current?.pause();
        setIsPaused(true);
    }, []);

    const resumeRecording = useCallback(() => {
        audioRecorderRef.current?.resume();
        setIsPaused(false);
    }, []);

    const stopAndSendRecording = useCallback(async () => {
        // Collapse any open panels when sending
        if (onSendStart) {
            onSendStart();
        }
        if (!isRecording || !audioRecorderRef.current) {
            return null;
        }
        const recorder = audioRecorderRef.current;
        audioRecorderRef.current = null;
        setIsRecording(false);
        setIsPaused(false);
        let liveResult = { text: "", authoritative: false };
        if (liveSessionRef.current) {
            try {
                liveResult = (await liveSessionRef.current.stop()) || liveResult;
            } catch {
                // overlay-only; batch transcription still runs
            }
            closeLiveSession();
        }
        const blob = await recorder.stop();
        await sendForTranscription(
            blob,
            {
                name,
                gender,
                dob,
                templateKey: template?.template_key,
                noteId,
            },
            liveResult,
        );
        return blob;
    }, [
        isRecording,
        sendForTranscription,
        name,
        gender,
        dob,
        template,
        noteId,
        onSendStart,
        closeLiveSession,
    ]);

    const resetRecording = useCallback(() => {
        if (isRecording && audioRecorderRef.current) {
            audioRecorderRef.current.stop().catch(() => {});
            audioRecorderRef.current = null;
        }
        resetRecordingState();
    }, [isRecording, resetRecordingState]);

    const toggleAmbientMode = useCallback(async () => {
        const newValue = !isAmbient;
        setIsAmbient(newValue);
        try {
            await settingsService.saveAmbientMode(newValue);
        } catch (error) {
            console.error("Failed to save ambient mode setting:", error);
        }
    }, [isAmbient]);

    // Handle audio file drop
    const handleAudioDrop = useCallback(
        async (file) => {
            if (!file.type.startsWith("audio/")) {
                return false;
            }

            return sendForTranscription(file, {
                name,
                gender,
                dob,
                templateKey: template?.template_key,
                noteId,
            });
        },
        [sendForTranscription, name, gender, dob, template, noteId],
    );

    return {
        // State
        isAmbient,
        requireConsent,
        isRecording,
        isPaused,
        timer,
        isLoading: isTranscribing,
        sendError,
        liveTranscript,
        liveError,

        // Actions
        startRecording,
        pauseRecording,
        resumeRecording,
        stopAndSendRecording,
        resetRecording,
        toggleAmbientMode,
        handleAudioDrop,
        retrySend,
        downloadLastRecording,
        dismissSendError,
    };
};

// The actual UI is in ScribePillBox and the panels
const Scribe = ({ children }) => {
    return children || null;
};

export default Scribe;
