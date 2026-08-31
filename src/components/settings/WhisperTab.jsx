import {
    Box,
    HStack,
    Input,
    InputGroup,
    NativeSelect,
    Spinner,
    Text,
    VStack,
} from "@chakra-ui/react";
import { Tooltip } from "@/components/ui/tooltip";
import { CheckCircleIcon } from "../common/icons";
import { applyAsrProviderDefaults } from "../../utils/aiProviders";

const WhisperTab = ({
    config,
    handleConfigChange,
    whisperModelOptions = [],
    whisperModelListAvailable = false,
    whisperModelsLoading = false,
    urlStatus = { whisper: false },
    asrProviders = [],
}) => {
    const provider = config?.ASR_PROVIDER || "openai_compatible";
    const isExternalProvider = provider !== "local";
    const modelValue = config?.ASR_MODEL || config?.WHISPER_MODEL || "";
    const selectedAsr = asrProviders.find((item) => item.id === provider);
    const urlPlaceholder =
        selectedAsr?.placeholder_url || "https://asr.example.com";

    const updateModel = (value) => {
        handleConfigChange("ASR_MODEL", value);
        handleConfigChange("WHISPER_MODEL", value);
    };

    const updateKey = (value) => {
        handleConfigChange("ASR_KEY", value);
        handleConfigChange("WHISPER_KEY", value);
    };

    return (
        <VStack gap={4} align="stretch" dir="rtl">
            <Box>
                <Text fontSize="md" fontWeight="bold">
                    ASR؛ تشخیص خودکار گفتار
                </Text>
                <Text fontSize="sm" color="overlay0">
                    سرویس پیاده‌سازی صدای ضبط‌شده را انتخاب و پیکربندی کنید.
                </Text>
            </Box>

            <VStack gap={3} align="stretch">
                <Box>
                    <Text fontSize="sm" mb="1" fontWeight="bold">
                        ارائه‌دهنده ASR
                    </Text>
                    <NativeSelect.Root>
                        <NativeSelect.Field
                            size="sm"
                            value={provider}
                            onChange={(event) =>
                                applyAsrProviderDefaults(
                                    event.target.value,
                                    handleConfigChange,
                                )
                            }
                            className="input-style"
                        >
                            {(asrProviders.length
                                ? asrProviders
                                : [
                                      { id: "local", name_fa: "مدل محلی؛ Whisper.cpp، Parakeet یا Shenava" },
                                      { id: "openai_compatible", name_fa: "سرویس سازگار با OpenAI" },
                                      { id: "openai", name_fa: "OpenAI Audio" },
                                      { id: "whispercpp", name_fa: "سرور Whisper.cpp" },
                                      { id: "speechmatics", name_fa: "Speechmatics؛ بلادرنگ" },
                                      { id: "fireworks", name_fa: "Fireworks AI ASR" },
                                  ]
                            ).map((item) => (
                                <option key={item.id} value={item.id}>
                                    {item.name_fa || item.name}
                                </option>
                            ))}
                        </NativeSelect.Field>
                        <NativeSelect.Indicator />
                    </NativeSelect.Root>
                    <Text fontSize="xs" color="overlay0" mt={1}>
                        ارائه‌دهنده، مدل و زبان ASR مستقل از مدل زبانی انتخاب می‌شوند.
                    </Text>
                </Box>

                <Box>
                    <Text fontSize="sm" mb="1" fontWeight="bold">
                        زبان گفتار
                    </Text>
                    <NativeSelect.Root>
                        <NativeSelect.Field
                            size="sm"
                            value={config?.ASR_LANGUAGE || config?.WHISPER_LANGUAGE || "auto"}
                            onChange={(event) => {
                                handleConfigChange("ASR_LANGUAGE", event.target.value);
                                handleConfigChange("WHISPER_LANGUAGE", event.target.value);
                            }}
                            className="input-style"
                        >
                            <option value="auto">تشخیص خودکار؛ فارسی و انگلیسی ترکیبی</option>
                            <option value="fa">فارسی</option>
                            <option value="en">انگلیسی</option>
                        </NativeSelect.Field>
                        <NativeSelect.Indicator />
                    </NativeSelect.Root>
                </Box>

                {["openai_compatible", "openai", "whispercpp", "fireworks", "speechmatics"].includes(
                    provider,
                ) && (
                    <Box>
                        <Tooltip content="نشانی پایه سرویس ASR را وارد کنید. برای Speechmatics می‌توانید از نشانی منطقه‌ای (مثلاً wss://us.rt.speechmatics.com/v2) یا global استفاده کنید.">
                            <Text fontSize="sm" mb="1" fontWeight="bold">
                                نشانی پایه سرویس ASR
                            </Text>
                        </Tooltip>
                        <InputGroup
                            size="sm"
                            endElement={
                                urlStatus.whisper ? (
                                    <Tooltip content="ارتباط با سرویس برقرار است.">
                                        <CheckCircleIcon color="successButton" />
                                    </Tooltip>
                                ) : undefined
                            }
                        >
                            <Input
                                type="url"
                                data-ltr="true"
                                dir="ltr"
                                value={config?.ASR_BASE_URL || config?.WHISPER_BASE_URL || ""}
                                onChange={(event) => {
                                    handleConfigChange("ASR_BASE_URL", event.target.value);
                                    handleConfigChange("WHISPER_BASE_URL", event.target.value);
                                }}
                                placeholder={urlPlaceholder}
                                className="input-style"
                            />
                        </InputGroup>
                    </Box>
                )}

                {provider === "speechmatics" && (
                    <Text fontSize="xs" color="overlay0">
                        Speechmatics در حالت بلادرنگ از شناسایی خودکار زبان پشتیبانی نمی‌کند؛ حالت «تشخیص خودکار» در حالت زنده به فارسی (fa) نگاشت می‌شود. فایل‌های ضبط‌شده با API دسته‌ای (Batch) پردازش می‌شوند که خودکار تشخیص زبان را پشتیبانی می‌کند. API Keys محصول‌محور هستند؛ کلید Realtime (rt) برای زنده و کلید Batch برای فایل‌ها لازم است.
                    </Text>
                )}

                {provider === "speechmatics" && (
                    <>
                        <Box>
                            <Tooltip content="نشانی API دسته‌ای Speechmatics (فایل‌های ضبط‌شده). پیش‌فرض: https://eu1.asr.api.speechmatics.com/v2">
                                <Text fontSize="sm" mb="1" fontWeight="bold">
                                    نشانی API دسته‌ای (فایل‌ها) — اختیاری
                                </Text>
                            </Tooltip>
                            <Input
                                type="url"
                                data-ltr="true"
                                dir="ltr"
                                value={config?.ASR_BATCH_URL || ""}
                                onChange={(event) =>
                                    handleConfigChange("ASR_BATCH_URL", event.target.value)
                                }
                                placeholder={selectedAsr?.batch_placeholder_url || "https://eu1.asr.api.speechmatics.com/v2"}
                                className="input-style"
                            />
                        </Box>
                        <Box>
                            <Tooltip content="اگر کلید شما فقط برای Realtime (type=rt) ساخته شده، برای پردازش فایل‌ها یک کلید Batch (type=batch) جداگانه اینجا وارد کنید. در غیر این صورت همان کلید اصلی استفاده می‌شود.">
                                <Text fontSize="sm" mb="1" fontWeight="bold">
                                    کلید Batch API — اختیاری
                                </Text>
                            </Tooltip>
                            <Input
                                size="sm"
                                type="password"
                                dir="ltr"
                                data-ltr="true"
                                value={config?.ASR_BATCH_KEY || ""}
                                onChange={(event) =>
                                    handleConfigChange("ASR_BATCH_KEY", event.target.value)
                                }
                                placeholder="کلید Batch (type=batch)"
                                className="input-style"
                            />
                        </Box>
                    </>
                )}

                <Box>
                    <Tooltip content="مدلی را انتخاب کنید که برای پیاده‌سازی گفتار استفاده می‌شود.">
                        <Text fontSize="sm" mb="1" fontWeight="bold">
                            مدل ASR
                        </Text>
                    </Tooltip>

                    {provider === "speechmatics" ? (
                        <NativeSelect.Root>
                            <NativeSelect.Field
                                size="sm"
                                value={modelValue || "enhanced"}
                                onChange={(event) => updateModel(event.target.value)}
                                className="input-style"
                            >
                                <option value="enhanced">حالت پیشرفته؛ دقت بالاتر</option>
                                <option value="standard">حالت استاندارد؛ سرعت بالاتر</option>
                            </NativeSelect.Field>
                            <NativeSelect.Indicator />
                        </NativeSelect.Root>
                    ) : provider === "fireworks" && !whisperModelListAvailable ? (
                        <NativeSelect.Root>
                            <NativeSelect.Field
                                size="sm"
                                value={modelValue || "fireworks-asr-v2"}
                                onChange={(event) => updateModel(event.target.value)}
                                className="input-style"
                            >
                                <option value="fireworks-asr-v2">Fireworks ASR v2 (زنده)</option>
                                <option value="fireworks-asr-large">Fireworks ASR Large (زنده)</option>
                                <option value="whisper-v3-turbo">Whisper v3 Turbo (دسته‌ای)</option>
                                <option value="whisper-v3">Whisper v3 (دسته‌ای)</option>
                            </NativeSelect.Field>
                            <NativeSelect.Indicator />
                        </NativeSelect.Root>
                    ) : whisperModelsLoading ? (
                        <HStack gap="2">
                            <Spinner size="sm" />
                            <Text fontSize="sm" color="overlay0">
                                در حال دریافت فهرست مدل‌ها…
                            </Text>
                        </HStack>
                    ) : whisperModelListAvailable && whisperModelOptions.length > 0 ? (
                        <NativeSelect.Root>
                            <NativeSelect.Field
                                size="sm"
                                value={modelValue}
                                onChange={(event) => updateModel(event.target.value)}
                                className="input-style"
                            >
                                <option value="">انتخاب مدل ASR</option>
                                {whisperModelOptions.map((model) => (
                                    <option key={model} value={model}>
                                        {model}
                                    </option>
                                ))}
                            </NativeSelect.Field>
                            <NativeSelect.Indicator />
                        </NativeSelect.Root>
                    ) : (
                        <Input
                            size="sm"
                            dir="ltr"
                            data-ltr="true"
                            placeholder={
                                provider === "local"
                                    ? "whisper-large-v3-turbo-q5_0"
                                    : "whisper-1"
                            }
                            value={modelValue}
                            onChange={(event) => updateModel(event.target.value)}
                            className="input-style"
                        />
                    )}
                </Box>

                {isExternalProvider && (
                    <Box>
                        <Tooltip content="کلید در پیکربندی رمزگذاری‌شده ذخیره و هنگام نمایش پوشانده می‌شود.">
                            <Text fontSize="sm" mb="1" fontWeight="bold">
                                کلید API
                            </Text>
                        </Tooltip>
                        <Input
                            size="sm"
                            type="password"
                            dir="ltr"
                            data-ltr="true"
                            value={config?.ASR_KEY || config?.WHISPER_KEY || ""}
                            onChange={(event) => updateKey(event.target.value)}
                            placeholder={
                                provider === "speechmatics"
                                    ? "کلید Speechmatics"
                                    : "کلید API، در صورت نیاز"
                            }
                            className="input-style"
                        />
                    </Box>
                )}
            </VStack>
        </VStack>
    );
};

export default WhisperTab;
