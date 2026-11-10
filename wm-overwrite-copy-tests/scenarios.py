"""The four experiment loops, keeping only one image pair in memory."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from attacks import Attack, Estimator, apply_attack, estimate_residual, forge
from errors import ExperimentError
from images import Example, load_image, save_example
from metrics import Context, Recorder, psnr
from settings import Config
from wm import TrustMarkWatermark, VideoSealWatermark, Watermark, payload


@dataclass(frozen=True, slots=True)
class ImageCase:
    index: int
    source: Path
    target: Path
    original: Image.Image


class Experiment:
    def __init__(self, config: Config, recorder: Recorder, trustmark: TrustMarkWatermark) -> None:
        self.config = config
        self.recorder = recorder
        self.trustmark = trustmark
        self.videoseal: VideoSealWatermark | None = None

    def message(self, case: ImageCase, method: Watermark, ordinal: int = 1) -> str:
        stream = (10 if method.name == "trustmark" else 20) + ordinal
        return payload(method.payload_length, (self.config.seed, case.index, stream))

    def observe(self, case: ImageCase, spec: tuple[Watermark, str, str, str], image: Image.Image) -> None:
        method, condition, reference_name, reference_bits = spec
        context = Context(
            scenario=self.config.scenario.value, image_index=case.index,
            source_file=case.source.name, target_file=case.target.name,
            condition=condition, method=method.name, reference=reference_name,
        )
        self.recorder.record(context, (method.decode(image), reference_bits), psnr(case.original, image))

    def example(self, case: ImageCase, label: str, images: Example) -> None:
        if case.index < 3:
            folder = self.config.output_dir / "examples" / f"{case.index:03d}"
            save_example(folder, label, images)

    def baseline(self, case: ImageCase) -> None:
        if self.videoseal is None:
            raise ExperimentError("VideoSeal required for baseline")
        methods: tuple[Watermark, ...] = (self.trustmark, self.videoseal)
        for method in methods:
            bits = self.message(case, method)
            self.observe(case, (method, "unmarked_control", "payload", bits), case.original)
            marked = method.embed(case.original, bits)
            for attack in Attack:
                attacked = apply_attack(marked, attack)
                self.observe(case, (method, attack.value, "payload", bits), attacked)
                if attack is Attack.JPEG80:
                    self.example(case, method.name, Example(case.original, marked, attacked))

    def overwrite(self, case: ImageCase) -> None:
        first_bits = self.message(case, self.trustmark)
        second_bits = self.message(case, self.trustmark, 2)
        if second_bits == first_bits:
            second_bits = second_bits[:-1] + str(1 - int(second_bits[-1]))
        first = self.trustmark.embed(case.original, first_bits)
        second = self.trustmark.embed(first, second_bits)
        # Decode the same final image once; score it independently against P1/P2.
        decoded = self.trustmark.decode(second)
        quality = psnr(case.original, second)
        for label, bits in (("P1", first_bits), ("P2", second_bits)):
            context = Context(
                scenario=self.config.scenario.value, image_index=case.index,
                source_file=case.source.name, target_file=case.target.name,
                condition="trustmark_then_trustmark", method="trustmark", reference=label,
            )
            self.recorder.record(context, (decoded, bits), quality)
        self.example(case, "overwrite", Example(case.original, first, second))

    def cross_method(self, case: ImageCase) -> None:
        if self.videoseal is None:
            raise ExperimentError("VideoSeal required for cross_method")
        orders: tuple[tuple[Watermark, Watermark], ...] = (
            (self.trustmark, self.videoseal), (self.videoseal, self.trustmark)
        )
        for first_method, second_method in orders:
            first_bits = self.message(case, first_method)
            second_bits = self.message(case, second_method, 2)
            first = first_method.embed(case.original, first_bits)
            second = second_method.embed(first, second_bits)
            condition = f"{first_method.name}_then_{second_method.name}"
            self.observe(case, (first_method, condition, "first_mark", first_bits), second)
            self.observe(case, (second_method, condition, "second_mark", second_bits), second)
            self.example(case, condition, Example(case.original, first, second))

    def copy_attack(self, case: ImageCase) -> None:
        bits = self.message(case, self.trustmark)
        marked = self.trustmark.embed(case.original, bits)
        target = load_image(case.target)
        target_case = ImageCase(case.index, case.source, case.target, target)
        self.observe(target_case, (self.trustmark, "unmarked_control", "donor_payload", bits), target)
        for estimator in Estimator:
            residual = estimate_residual(marked, estimator)
            for strength in (1.0, 2.0, 3.0):
                forged = forge(target, residual, strength)
                condition = f"{estimator.value}_s{strength:.1f}"
                self.observe(target_case, (self.trustmark, condition, "donor_payload", bits), forged)
                if strength == 2.0:
                    self.example(target_case, estimator.value, Example(target, marked, forged))
        if case.index < 3:
            folder = self.config.output_dir / "examples" / f"{case.index:03d}"
            case.original.save(folder / "donor_original.png")
