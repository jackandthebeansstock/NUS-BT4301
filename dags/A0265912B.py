from airflow.decorators import dag, task
from airflow.utils.dates import datetime, timedelta
from airflow.models import Variable
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import matplotlib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
matplotlib.use("Agg")
import seaborn as sns

AIRFLOW_HOME = Variable.get("AIRFLOW_HOME", os.getcwd())
OUTPUT_DIR = os.path.join(AIRFLOW_HOME, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

@dag(
    dag_id="assignment3",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
    tags=["assignment3"],
)
def assignment3_dag():
    """1.a Instantiating the DAG"""
    @task
    def extract_sales_data():
        """1.b Downloading the Kaggle dataset, extracting relevant columns, and saving the CSV"""
        path = os.path.join(AIRFLOW_HOME, "kaggle_dataset")
        os.makedirs(path, exist_ok=True)

        df = pd.read_csv(os.path.join(AIRFLOW_HOME,"supermarket_sales.csv"))

        columns = ["Invoice ID", "Branch", "City", "Customer type", "Product line", "Total", "Date"]
        df = df[columns]

        saved_df_path = os.path.join(OUTPUT_DIR, "saved_df.csv")
        df.to_csv(saved_df_path, index=False)

        return saved_df_path

    @task(multiple_outputs=True)
    def transform_sales_data(saved_df_path: str):
        #2a
        
        df = pd.read_csv(saved_df_path)

        ttl_sales_by_city = df.groupby('City')['Total'].sum().to_dict()
        avg_ttl_by_product_line = df.groupby('Product line')['Total'].mean().to_dict()

        ttl_sales = df['Total'].sum()
        proportions_by_product_line = (df.groupby('Product line')['Total'].sum() / ttl_sales * 100).to_dict()

        sales_summary = {
            "Total Sales By City": ttl_sales_by_city,
            "Avg Total By Product Line": avg_ttl_by_product_line,
            "Proportions By Product Line": proportions_by_product_line
        }

        #2b

        sales_summary_path = os.path.join(OUTPUT_DIR,"sales_summary.json")
        with open(sales_summary_path, "w") as json_file:
            json.dump(sales_summary, json_file)
        

        #2c

        city_summary = {
            city: {
                "Total Sales Per Branch": df[df["City"] == city].groupby("Branch")["Total"].sum().to_dict(),
                "Total Sales Per Product Line": df[df["City"] == city].groupby("Product line")["Total"].sum().to_dict()
            }
            for city in df["City"].unique()
        }

        #2d

        city_summary_path = os.path.join(OUTPUT_DIR, "city_summary.json")
        with open(city_summary_path, "w") as json_file:
            json.dump(city_summary, json_file)

        return {"sales_summary_path": sales_summary_path, "city_summary_path": city_summary_path}



    @task
    def generate_sales_report(sales_summary_path):
        with open(sales_summary_path, "r") as json_file:
            sales_summary = json.load(json_file)
        
        ttl_sales_by_city = sales_summary["Total Sales By City"]
        avg_ttl_by_product_line = sales_summary["Avg Total By Product Line"]
        proportions_by_product_line = sales_summary["Proportions By Product Line"]
        
        # Generate all charts and collect their paths
        image_paths = [
            _create_city_sales_chart(ttl_sales_by_city),
            _create_product_line_avg_chart(avg_ttl_by_product_line),
            _create_product_line_pie_chart(proportions_by_product_line)
        ]
        
        # Create PDF with all charts
        pdf_path = os.path.join(OUTPUT_DIR, "A0265912B_sales_report.pdf")
        _create_pdf_with_images(image_paths, pdf_path)
        
        return pdf_path


    def _create_city_sales_chart(ttl_sales_by_city):
        plt.figure(figsize=(8, 5))
        plt.bar(ttl_sales_by_city.keys(), ttl_sales_by_city.values(), color="lightblue")
        plt.title("Total Sales by City")
        plt.xlabel("City")
        plt.ylabel("Total Sales")
        
        path = os.path.join(os.getcwd(), "total_sales_by_city.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        
        return path


    def _create_product_line_avg_chart(avg_ttl_by_product_line):
        plt.figure(figsize=(8, 5))
        plt.bar(avg_ttl_by_product_line.keys(), avg_ttl_by_product_line.values(), color="orange")
        plt.title("Average Total by Product Line")
        plt.xlabel("Product Line")
        plt.ylabel("Average Sales Total")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        
        path = os.path.join(os.getcwd(), "avg_total_by_product_line.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        
        return path


    def _create_product_line_pie_chart(proportions_by_product_line):
        plt.figure(figsize=(6, 6))
        product_lines = list(proportions_by_product_line.keys())
        sales_values = list(proportions_by_product_line.values())
        plt.pie(sales_values, labels=product_lines, autopct="%1.1f%%", startangle=140)
        plt.title("Proportion of Sales by Product Line")
        
        path = os.path.join(os.getcwd(), "proportion_by_product_line.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        
        return path

    @task
    def get_city_list(city_summary_path: str):
        with open(city_summary_path, "r") as f:
            city_summary = json.load(f)
        return list(city_summary.keys())

    @task
    def generate_city_report(city, city_summary_path):

        with open(city_summary_path, "r") as json_file:
            city_summary = json.load(json_file)
        
        city_data = city_summary[city]
        
        # Generate all charts for this city
        image_paths = [
            _create_city_branch_sales_chart(city, city_data),
            _create_city_product_line_chart(city, city_data)
        ]
        
        # Create PDF with the charts
        pdf_path = os.path.join(OUTPUT_DIR, f"A0265912B_{city}_sales_report.pdf")
        _create_pdf_with_images(image_paths, pdf_path)
        
        return pdf_path


    def _create_city_branch_sales_chart(city, city_data):
        plt.figure(figsize=(8, 5))
        sns.barplot(
            x=list(city_data["Total Sales Per Branch"].keys()),
            y=list(city_data["Total Sales Per Branch"].values()),
            palette="Blues"
        )
        plt.title(f"Total Sales by Branch - {city}")
        plt.xlabel("Branch")
        plt.ylabel("Total Sales")
        plt.xticks(rotation=15, ha="right")
        
        path = os.path.join(os.getcwd(), f"{city}_total_sales_by_branch.png")
        plt.savefig(path)
        plt.close()
        
        return path


    def _create_city_product_line_chart(city, city_data):
        plt.figure(figsize=(8, 5))
        sns.barplot(
            x=list(city_data["Total Sales Per Product Line"].keys()),
            y=list(city_data["Total Sales Per Product Line"].values()),
            palette="Blues"
        )
        plt.title(f"Total Sales by Product Line - {city}")
        plt.xlabel("Product Line")
        plt.ylabel("Total Sales")
        plt.xticks(rotation=15, ha="right")
        
        path = os.path.join(os.getcwd(), f"{city}_total_sales_by_product_line.png")
        plt.savefig(path)
        plt.close()
        
        return path


    def _create_pdf_with_images(image_paths, pdf_path):
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        margin = 50
        image_height = 400
        current_height = height - margin
        
        for image_path in image_paths:
            if current_height < margin + image_height:
                c.showPage()
                current_height = height - margin
            
            c.drawImage(image_path, margin, current_height - image_height, width=500, height=image_height)
            current_height -= (image_height + margin)
        
        c.save()



    saved_df_path = extract_sales_data()
    
    data = transform_sales_data(saved_df_path)
    sales_summary_path = data["sales_summary_path"]
    city_summary_path = data["city_summary_path"]
    
    # Generate overall sales report
    sales_summary_report = generate_sales_report(sales_summary_path)
    
    # Generate city-specific reports
    city_list = get_city_list(city_summary_path)
    city_reports = generate_city_report.partial(
        city_summary_path=city_summary_path
    ).expand(city=city_list)
    
    # Define task dependencies
    saved_df_path >> (sales_summary_path, city_summary_path)
    sales_summary_path >>sales_summary_report
    sales_summary_report >> city_reports
    city_summary_path >> city_reports



assignment3_dag()