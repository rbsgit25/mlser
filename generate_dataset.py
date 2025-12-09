from main import generate_dataset


if __name__ == "__main__":
	df = generate_dataset(n=2000)
	print("Generated", len(df), "rows -> synthetic_ecom.csv")