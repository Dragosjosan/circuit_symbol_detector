import sys
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from loguru import logger

logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",
)

SYMBOL_TO_FIND = Path(__file__).parent / "resonator.png"
DRAWINGS_TO_DETECT_IN = Path(__file__).parent / "data"
RESULT_FOLDER = Path(__file__).parent / "results"
ALLOWED_DRAWING_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif")


class TemplateMatchError(Exception):
    pass


def check_if_input_files_exist() -> None:
    if not all((SYMBOL_TO_FIND.exists(), SYMBOL_TO_FIND.is_file())):
        logger.error(f"Symbol to find not found, or is not a file: {SYMBOL_TO_FIND}")
        raise FileNotFoundError(f"Symbol to find not found: {SYMBOL_TO_FIND}")
    if not all((DRAWINGS_TO_DETECT_IN.exists(), DRAWINGS_TO_DETECT_IN.is_dir())):
        logger.error(
            f"Drawings folder to detect in not found, or is not a directory: {DRAWINGS_TO_DETECT_IN}"
        )
        raise FileNotFoundError(
            f"Drawings folder to detect in not found: {DRAWINGS_TO_DETECT_IN}"
        )

    RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

    files = list(DRAWINGS_TO_DETECT_IN.iterdir())
    if not files:
        logger.error(f"No files found in drawings directory: {DRAWINGS_TO_DETECT_IN}")
        raise FileNotFoundError(
            f"No files found in drawings directory: {DRAWINGS_TO_DETECT_IN}"
        )

    valid_files = [
        file
        for file in files
        if file.is_file() and file.suffix.lower() in ALLOWED_DRAWING_FILE_EXTENSIONS
    ]
    if not valid_files:
        logger.error(
            f"No valid image files found in drawings directory: {DRAWINGS_TO_DETECT_IN}"
        )
        raise FileNotFoundError(
            f"No valid image files found in drawings directory: {DRAWINGS_TO_DETECT_IN}"
        )


def extract_width_and_height(template: np.ndarray) -> Tuple[int, int]:
    height, width = template.shape
    return width, height


def get_files_to_detect_in() -> List[Path]:
    return [
        file
        for file in DRAWINGS_TO_DETECT_IN.iterdir()
        if file.is_file() and file.suffix.lower() in ALLOWED_DRAWING_FILE_EXTENSIONS
    ]


def process_files(
    files: List[Path], template: np.ndarray, width: int, height: int
) -> None:
    for file in files:
        try:
            logger.info(f"Processing file: {file.name}")
            main_image = cv.imread(str(file))

            if main_image is None:
                logger.error(f"Could not read image file: {file}")
                continue

            logger.debug("Converting main image to grayscale")
            main_image_gray = convert_to_grayscale(image=main_image)

            logger.info("Finding matches across rotations")
            found_images = find_symbol_in_image_for_multiple_angles(
                image=main_image_gray, template=template
            )

            logger.info("Applying non-maximum suppression")
            final_matches = non_max_suppression(
                matches=found_images, width=width, height=height
            )

            logger.info("Drawing results")
            save_image_results(
                result_image=main_image,
                matches=final_matches,
                file_path=file,
                width=width,
                height=height,
            )

        except TemplateMatchError as e:
            logger.error(f"Error processing file {file.name}: {str(e)}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing file {file.name}: {str(e)}")
            continue


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    rotation_matrix = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv.warpAffine(
        image, rotation_matrix, (width, height), borderValue=(255, 255, 255)
    )
    return rotated_image


def non_max_suppression(
    matches: List[Tuple[Tuple[int, int], float, float]],
    width: int,
    height: int,
    overlap_threshold: float = 0.3,
) -> List[Tuple[Tuple[int, int], float, float]]:
    if not matches:
        return []

    matches = sorted(matches, key=lambda x: x[2], reverse=True)
    keep = []

    while matches:
        current = matches.pop(0)
        keep.append(current)

        remaining = []
        current_rect = (current[0][0], current[0][1], width, height)

        for match in matches:
            match_rect = (match[0][0], match[0][1], width, height)

            x1 = max(current_rect[0], match_rect[0])
            y1 = max(current_rect[1], match_rect[1])
            x2 = min(current_rect[0] + current_rect[2], match_rect[0] + match_rect[2])
            y2 = min(current_rect[1] + current_rect[3], match_rect[1] + match_rect[3])

            if x2 < x1 or y2 < y1:
                overlap = 0
            else:
                intersection = (x2 - x1) * (y2 - y1)
                current_area = current_rect[2] * current_rect[3]
                match_area = match_rect[2] * match_rect[3]
                union = current_area + match_area - intersection
                overlap = intersection / union

            if overlap < overlap_threshold:
                remaining.append(match)

        matches = remaining

    return keep


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    # Check if image is not None before converting
    if image is None:
        raise TemplateMatchError("Cannot convert None image to grayscale")
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def find_symbol_in_image_for_multiple_angles(
    image: np.ndarray, template: np.ndarray
) -> List[Tuple[Tuple[int, int], float, float]]:
    all_matches = []
    angles = range(0, 360, 15)
    threshold = 0.7

    for angle in angles:
        try:
            rotated_template = rotate_image(template, angle)
            res = cv.matchTemplate(image, rotated_template, cv.TM_CCOEFF_NORMED)
            locations = np.where(res >= threshold)
            for pt in zip(*locations[::-1]):
                all_matches.append((pt, angle, res[pt[1], pt[0]]))
        except Exception as e:
            logger.warning(f"Failed processing angle {angle}: {str(e)}")
            continue
    return all_matches


def save_image_results(
    result_image: np.ndarray, matches: List, file_path: Path, width: int, height: int
) -> None:
    result_image = result_image.copy()
    for pt, angle, conf in matches:
        cv.rectangle(result_image, pt, (pt[0] + width, pt[1] + height), (0, 0, 255), 2)
        cv.putText(
            result_image,
            f"{angle}°, {conf:.2f}",
            (pt[0], pt[1] - 5),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    result_path = RESULT_FOLDER / f"result_{file_path.stem}.png"
    cv.imwrite(str(result_path), result_image)
    logger.info(f"Saved result to: {result_path}")


def main() -> None:
    try:
        logger.info("Starting template matching")
        logger.info("Checking if input files exist")
        check_if_input_files_exist()

        logger.info("Loading template image")
        template = cv.imread(str(SYMBOL_TO_FIND), cv.IMREAD_GRAYSCALE)
        if template is None:
            raise TemplateMatchError(f"Template could not be read: {SYMBOL_TO_FIND}")

        logger.debug("Extracting width and height from template")
        width, height = extract_width_and_height(template=template)

        logger.info("Extracting the input files to detect the symbol in")
        files_to_detect_in = get_files_to_detect_in()
        logger.debug(f"Found {len(files_to_detect_in)} files to detect the symbol in")
        logger.debug(f"Files to detect the symbol in: {files_to_detect_in} \n")

        logger.info("Processing files")
        process_files(
            files=files_to_detect_in, template=template, width=width, height=height
        )
        logger.info("Template matching completed successfully!")

    except TemplateMatchError as e:
        logger.exception(f"Template matching error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
