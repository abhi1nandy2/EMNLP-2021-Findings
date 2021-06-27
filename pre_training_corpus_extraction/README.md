## Extraction of pre-training corpus

## Files

1. `extract_1.py, extract_2.py, extract_3.py, extrat_4.py` - In order to get access to any PDF, there are 4 hops of links required from the home page of http://www.manualsonline.com/, hence 4 python files are required to be run one after another. In the end, a `json` file containing links to EManual PDFs is produced in the following format -
```
{
  "All_Manuals": [
    {   # this is an entry of the list within 'All_Manuals' key.
    	# each such entry corresponds to a brand-category pair. There are 38611 such pairs 
      "brand": "twelvevolt",
      "category": "Air Compressor Manuals",
      "brand_category_url": "http://powertool.manualsonline.com/manuals/mfg/twelvevolt/twelvevolt_air_compressor_product_list.html",
      "models": [
      	# this is a list of models within each brand-category pair. Each model is represented by a 
      	# dictionary containing info on the model and the download link of the pdf.
      	# This list is empty when there are no models corresponding to that brand-category pair.
      	# Within a brand-category, there might be duplicate links of some models. That is taken care of
        {
          "title": "12Volt Air Compressor 12V6CF",
          "desc": "12Volt Air Compressor User Manual",
          "num_pages": "Pages:  16",
          "model_url": "http://powertool.manualsonline.com/manuals/mfg/twelvevolt/12v6cf.html",
          "download_link": "http://pdfstream.manualsonline.com/6/6e205ecf-8a6b-4e30-8ffb-08780ad42874.pdf"
        }
      ]
    },
    .
    .
    .
  ]
}
```
2. `extract_text-from_pdf.py` - Given a list of links, this file, when run, extracts text from each PDF in the way the text is organized in a PDF (PDF -> block -> span), and hen stores it in separate locations
3. `merge_all_emanual_txt.py` - Merging all separate text files for different PDFs into one single text file, removing non-unicode and non-ASCII characters.
