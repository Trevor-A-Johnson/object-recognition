from extractData import extract_images

if __name__ == '__main__':
    print("Loading images...")
    images = extract_images()
    print(f"Total number of images extracted: {len(images)}")
