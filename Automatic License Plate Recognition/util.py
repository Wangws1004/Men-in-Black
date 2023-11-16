import string
import easyocr
import cv2
import imutils
import numpy as np
from scipy.interpolate import interp1d
import ast
import re
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm



# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}
hangul_characters = {'가', '나', '다', '라', '마', '거', '너', '더', '러', '머', '버', '서', '어', '저', '고', '노', '도', '로', '모', '보',
                     '소', '오', '조', '구', '누', '두', '루', '무', '부', '수', '우', '주', '그', '느', '드', '르', '므', '브', '스', '으',
                     '즈', '허', '하', '호'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                # print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                        'license_plate' in results[frame_nmr][car_id].keys() and \
                        'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required Korean format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """

    # Check for format: Hangul-Hangul-Number-Number-Number-Number-Hangul
    if len(text) == 7 and \
            text[0] in hangul_characters and \
            text[1] in hangul_characters and \
            text[2:6].isdigit() and \
            text[6] in hangul_characters:
        return True

    # Check for format: Number-Number(-Number)-Hangul-Number-Number-Number-Number
    elif len(text) in [7, 8] and \
            text[:-5].isdigit() and \
            text[-5] in hangul_characters and \
            text[-4:].isdigit():
        return True

    elif len(text) == 7 and \
            (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
            (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
            (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
            (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
            (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
            (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
            (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True

    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def clean_text(text):
    """
    Clean the text by removing special characters, keeping only letters, numbers, and Hangul.

    Args:
        text (str): Text detected by OCR.

    Returns:
        str: Cleaned text.
    """
    # Define a regex pattern that matches everything except letters, numbers, and Hangul
    pattern = '[^A-Za-z0-9가-힣]'

    # Remove special characters using regex sub
    cleaned_text = re.sub(pattern, '', text)

    return cleaned_text.upper()  # Convert to upper case and return

# Usage in your read_license_plate function
def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        # Clean the text
        cleaned_text = clean_text(text)
        print('text:', text)
        print('cleaned text:', cleaned_text)
        if license_complies_format(cleaned_text):
            return cleaned_text, score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


def interpolate_bounding_boxes(data):
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:

        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
        print(frame_numbers_, car_id)

        # Filter data for a specific car ID
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i - 1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0,
                                           kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['car_id'] = str(car_id)
            row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if
                                int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][
                    0]
                row['license_plate_bbox_score'] = original_row[
                    'license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row[
                    'license_number_score'] if 'license_number_score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=7, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # -- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # -- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # -- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # -- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def process_and_visualize_license_plate(license_plate_crop):
    if license_plate_crop.size > 0:
        # Convert to grayscale
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        license_plate_crop_clahe = clahe.apply(license_plate_crop_gray)

        # Apply Gaussian blur to reduce noise
        license_plate_crop_blur = cv2.GaussianBlur(license_plate_crop_clahe, (5, 5), 0)

        # Apply Canny edge detection
        license_plate_crop_canny = cv2.Canny(license_plate_crop_blur, 75, 200)

        # Find contours and identify the one that most likely represents the license plate
        cnts = cv2.findContours(license_plate_crop_canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]  # Sort by area, get the largest

        screenCnt = None
        for c in cnts:
            # Approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)

            # The contour is likely to be the license plate if it's rectangular (has four points)
            if len(approx) == 4:
                screenCnt = approx
                break

        # Perspective transformation if a rectangular contour is found
        if screenCnt is not None:
            # Coordinates of the found contour
            pts = screenCnt.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")

            # The top-left point has the smallest sum, whereas
            # the bottom-right point has the largest sum
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]

            # Compute the difference between the points:
            # the top-right point will have the minumum difference,
            # the bottom-left will have the maximum difference
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            # Transform perspective
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = max(int(widthA), int(widthB))
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = max(int(heightA), int(heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(license_plate_crop, M, (maxWidth, maxHeight))

            return warp

        return license_plate_crop_clahe  # Return the CLAHE image if no contour is found


def visualize(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Path to the Korean font
    korean_font_path = './fonts/NanumGothic-ExtraBold.ttf'
    korean_font_size = 80

    # Check if the font file exists
    if not os.path.exists(korean_font_path):
        raise FileNotFoundError(f"Korean font file not found at {korean_font_path}")

    korean_font = ImageFont.truetype(korean_font_path, korean_font_size)

    results = pd.read_csv(f'./sample/{video_name}_interpolated.csv')

    # load video
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(f'./sample/{video_name}_out.mp4', fourcc, fps, (width, height))

    license_plate = {}
    for car_id in np.unique(results['car_id']):
        max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
        license_plate[car_id] = {'license_crop': None,
                                 'license_plate_number': results[(results['car_id'] == car_id) &
                                                                 (results['license_number_score'] == max_)][
                                     'license_number'].iloc[0]}
        cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                                 (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
        ret, frame = cap.read()

        x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                                  (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[
                                              0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ',
                                                                                                                   ','))

        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 200 / (y2 - y1)), 200))

        license_plate[car_id]['license_crop'] = license_crop

    frame_nmr = -1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # read frames
    ret = True


    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            df_ = results[results['frame_nmr'] == frame_nmr]
            for row_indx in range(len(df_)):
                # draw car
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                    df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(
                        ' ', ','))
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                            line_length_x=200, line_length_y=200)

                # draw license plate
                x1, y1, x2, y2 = ast.literal_eval(
                    df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ',
                                                                                                            ' ').replace(
                        ' ', ','))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                # crop license plate
                license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']
                H, W, _ = license_crop.shape

                # Check if there is enough space to place the image
                available_space = int(car_y1) - 100  # Space above the car

                # If not enough space, resize the image to half its size
                if H > available_space:
                    new_height = H // 2
                    new_width = W // 2
                    license_crop = cv2.resize(license_crop, (new_width, new_height))
                    H, W, _ = license_crop.shape

                # Adjust the position to reduce gap
                placement_y = int(car_y1) - H  # Adjust the gap here

                try:
                    frame[placement_y:placement_y + H,
                    int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                    # Prepare for drawing text with PIL
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)

                    # Check if the text is in Korean
                    license_text = license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number']
                    is_korean = any("\uac00" <= char <= "\ud7a3" for char in license_text)

                    text_x = int((car_x2 + car_x1 - W) / 2)
                    text_y = placement_y - korean_font_size - 10  # Adjust y position for text

                    if is_korean:
                        draw.text((text_x, text_y), license_text, font=korean_font, fill=(0, 0, 0))
                    else:
                        draw.text((text_x, text_y), license_text, font=ImageFont.truetype("Arial.ttf", 40),
                                  fill=(0, 0, 0))

                    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                except Exception as e:
                    print("Error:", e)
                    pass

            out.write(frame)
            frame = cv2.resize(frame, (1280, 720))

            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)

    out.release()
    cap.release()