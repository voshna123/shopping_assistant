from langchain_ibm import WatsonxLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import pandas as pd

# Granite Model Configuration
MODEL_ID = "ibm/granite-3-8b-instruct"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
PROJECT_ID = "skills-network"

model_parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 5000,
    "repetition_penalty": 1,
}

# Initialize Granite LLM
llm = WatsonxLLM(
    model_id=MODEL_ID,
    url=WATSONX_URL,
    project_id=PROJECT_ID,
    params=model_parameters
)

# Define JSON output structure for structured response
class SerpAPIPromptResponse(BaseModel):
    refined_query: str = Field(description="Refined search query for the product search.")
    additional_info: str = Field(description="Additional adjectives summarized to be added to the search query.")

# JSON output parser
json_parser = JsonOutputParser(pydantic_object=SerpAPIPromptResponse)

# Granite LLM Prompt Template
granite_template = PromptTemplate(
    template="System: {system_prompt}\n{format_prompt}\nHuman: {user_prompt}\nAI:",
    input_variables=["system_prompt", "format_prompt", "user_prompt"]
)

# A function to retrieve the gl code using LLM for geo-specific search
def llm_generate_gl(location):
    """
    Use the LLM to determine the ISO 3166-1 alpha-2 country code (gl) from a location string.
    """
    system_prompt = (
        "You are an expert at identifying ISO 3166-1 alpha-2 country codes from locations. "
        "Output only the two-letter country code (e.g., 'US' for United States) for the provided location, "
        "without any additional text or explanations."
    )
    format_prompt = "Output the ISO 3166-1 alpha-2 country code only."
    user_prompt = f"Location: {location}"

    try:
        chain = granite_template | llm
        response = chain.invoke({
            "system_prompt": system_prompt,
            "format_prompt": format_prompt,
            "user_prompt": user_prompt
        })
        # print(f"LLM Response for gl code: {response}")  # Debug the raw response

        # Extract and validate the country code
        gl_code = response.strip()  # Remove whitespace
        if len(gl_code) == 2 and gl_code.isalpha():  # Ensure valid ISO 3166-1 alpha-2 code
            return gl_code.upper()
        else:
            print(f"Invalid country code returned: {gl_code}")
            return None  # Return None for invalid codes
    except Exception as e:
        print(f"Error in llm_generate_gl: {e}")
        return None



# Function to refine user input
def refine_query(user_input, location):
    """
    Use Granite LLM to refine user input for SerpAPI.

    Args:
        user_input (str): Raw input query from the user.

    Returns:
        dict: A structured response containing the refined query and additional adjectives.
    """
    system_prompt = "You are a highly skilled shopping assistant. Refine user queries into specific product searches."
    format_prompt = json_parser.get_format_instructions()
    chain = granite_template | llm | json_parser
    response = chain.invoke({
        "system_prompt": system_prompt,
        "format_prompt": format_prompt,
        "user_prompt": user_input
    })
    return response  # Directly return the parsed response

def generate_comparison_table(products, featured_results=None, nearby_results=None):
    """
    Generate a comparison table for products, including deals and additional shopping details.

    Args:
        products (list): List of main shopping results.
        featured_results (list): List of featured shopping results (optional).
        nearby_results (list): List of nearby shopping results (optional).

    Returns:
        tuple: A tuple containing the comparison table as HTML and a detailed summary.
    """
    # Combine all product results (main + featured) for comparison
    all_products = products[:10]
    if featured_results:
        all_products += featured_results[:5]  # Add top 5 featured results

    # Prepare product summaries
    product_summaries = [
        {
            "Name": f"<a href='{p.get('product_link', '#')}' target='_blank'>{p.get('title', 'N/A')}</a>",
            "Price Now": p.get("price", "N/A"),
            "Original Price": p.get("old_price", "N/A"),
            "Special Offer": ", ".join(p.get("extensions", [])) if p.get("extensions") else "No Discount",
            "Rating": f"{p.get('rating', 'N/A')} ⭐",
            "Reviews": f"{p.get('reviews', 'N/A')} Reviews",
            "Store": f"<img src='{p.get('source_icon', '#')}' alt='{p.get('source', 'N/A')}' style='height:20px;vertical-align:middle;'/> {p.get('source', 'N/A')}",
            "Delivery": p.get("delivery", "N/A"),
            "Image": f"<img src='{p.get('thumbnail', '#')}' alt='Product Image' style='height:50px;'/>"
        }
        for p in all_products
    ]

    # Prepare product summaries
    product_summaries_llm = [
        {
            "Name": p.get("title", "N/A"),
            "Price Now": p.get("price", "N/A"),
            "Original Price": p.get("old_price", "N/A"),
            "Special Offer": ", ".join(p.get("extensions", [])) if p.get("extensions") else "No Discount",
            "Rating": f"{p.get('rating', 'N/A')}",
            "Reviews": f"{p.get('reviews', 'N/A')} Reviews",
            "Store": f"<img src='{p.get('source_icon', '#')}' alt='{p.get('source', 'N/A')}' style='height:20px;vertical-align:middle;'/> {p.get('source', 'N/A')}",
            "Delivery": p.get("delivery", "N/A"),
            #"Image": f"<img src='{p.get('thumbnail', '#')}' alt='Product Image' style='height:50px;'/>"
        }
        for p in all_products
    ]


    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame(product_summaries)
    comparison_df_llm=pd.DataFrame(product_summaries_llm)

    if comparison_df.empty:
        return "<h3>No Products Found</h3>", "<h3>Summary Unavailable</h3><p>No product data available for analysis.</p>"

    # Generate a detailed summary using Granite LLM
    summary_prompt = (
        "You are an intelligent and detail-oriented shopping assistant. Analyze the following products and generate a detailed summary. "
        "Your response should include the following sections formatted as valid HTML:"
        "<h3>Best Value Product</h3>: Identify the product with the best price-to-value ratio."
        "<h3>Highest Rated Option</h3>: Highlight the product with the highest user rating."
        "<h3>Unique Features</h3>: List any unique or distinctive features of specific products."
        "<h3>Trade-offs and Comparisons</h3>: Discuss trade-offs between products, considering price, ratings, and reviews."
        "<h3>Conclusion and Suggestion</h3>: Provide a clear recommendation and reasoning for the best product or approach."
        "Use <ul> for lists and <li> for each item. Ensure all HTML tags are properly closed."
        "Here is the product information:\n"
        f"{comparison_df_llm.to_dict(orient='records')}")

    try:
        llm_result = llm.generate([summary_prompt])
        if not llm_result.generations or not llm_result.generations[0]:
            raise ValueError("LLM returned an empty response.")
        summary = llm_result.generations[0][0].text  # Extract the generated summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        summary = "<h3>Summary Unavailable</h3><p>An error occurred while generating the summary. Please try again later.</p>"

    # Ensure all sections are included in the summary
    required_sections = [
        "<h3>Best Value Product</h3>",
        "<h3>Highest Rated Option</h3>",
        "<h3>Unique Features</h3>",
        "<h3>Trade-offs and Comparisons</h3>",
        "<h3>Conclusion and Suggestion</h3>",
    ]
    for section in required_sections:
        if section not in summary:
            summary += f"\n{section}\n<p>Data unavailable for this section.</p>"

    # Convert the comparison DataFrame to an HTML table
    comparison_table_html = comparison_df.to_html(index=False, escape=False)  # Disable escaping for links and images

    return comparison_table_html, summary


# Example testing of both functionalities
if __name__ == "__main__":
    # Example input for query refinement
    user_input = "Find affordable jeans in Austin"
    location="Austin, Texas"
    refined_response = refine_query(user_input,location)
    print("Refined Query:", refined_response["refined_query"])
    print("Additional Info:", refined_response["additional_info"])

    # Example product data for comparison
    sample_products = [
        {
            "title": "Levi's 501 Original Jeans",
            "price": "$69.99",
            "source": "Amazon",
            "link": "https://www.amazon.com/example1",
            "rating": 4.5,
            "reviews": 120
        },
        {
            "title": "Wrangler Men's Jeans",
            "price": "$49.99",
            "source": "Walmart",
            "link": "https://www.walmart.com/example2",
            "rating": 4.3,
            "reviews": 98
        },
        {
            "title": "Lee Relaxed Fit Jeans",
            "price": "$39.99",
            "source": "Target",
            "link": "https://www.target.com/example3",
            "rating": 4.1,
            "reviews": 85
        },
        {
            "title": "Gap Straight Jeans",
            "price": "$59.99",
            "source": "Gap",
            "link": "https://www.gap.com/example4",
            "rating": 4.4,
            "reviews": 110
        },
        {
            "title": "Uniqlo Slim Jeans",
            "price": "$39.90",
            "source": "Uniqlo",
            "link": "https://www.uniqlo.com/example5",
            "rating": 4.6,
            "reviews": 150
        }
    ]

    # Generate comparison table and summary
    comparison_table, summary = generate_comparison_table(sample_products)
    print("Comparison Table:\n", comparison_table)
    print("Summary:\n", summary)