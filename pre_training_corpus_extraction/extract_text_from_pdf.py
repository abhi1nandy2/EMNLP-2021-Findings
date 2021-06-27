"""
This program extracts the text of an input PDF and writes it in a text file.
The input file name is provided as a parameter to this script (sys.argv[1])
The output file name is input-filename appended with ".txt".
Encoding of the text in the PDF is assumed to be UTF-8.
Change the ENCODING variable as required.
-------------------------------------------------------------------------------
"""
import fitz                 # this is PyMuPDF
import sys, json
from tqdm import tqdm
import requests
import os

proxy_dict = {'https':'', 'http':'', 'ftp':''}

ENCODING = "UTF-8"

# take first element for sort
def takeFirst(elem):
    return elem[0]

def SortBlocks(blocks, total_width, num_x_buckets = 5):
    '''
    Sort the blocks of a TextPage in ascending horizontal pixel order,
    then in ascending vertical pixel order.

    num_x_buckets - the page's width is divided into equal width buckets, such that,
    x-coordinate is measured in terms of the bucket it lies in. This helps in cases
    when there are small vetor images with paragraph wrapped around it. 
    '''

    sblocks = []
    for b in blocks:
        x0 = int(b["bbox"][0]+0.99999) # x coord in pixels
        unit_ = total_width/num_x_buckets
        x0 = str(int(x0/unit_))
        y0 = str(int(b["bbox"][1]+0.99999)).rjust(4,"0") # y coord in pixels
        sortkey = x0 + y0                                # = "yx"
        sblocks.append([sortkey, b])
    # print(sblocks[0])
    sblocks.sort(key = takeFirst)
    return [b[1] for b in sblocks] # return sorted list of blocks

def SortLines(lines):
    ''' Sort the lines of a block in ascending vertical direction. See comment
    in SortBlocks function.
    '''
    slines = []
    for l in lines:
        y0 = str(int(l["bbox"][1] + 0.99999)).rjust(4,"0")
        slines.append([y0, l])
    slines.sort(key = takeFirst)
    return [l[1] for l in slines]

def SortSpans(spans):
    ''' Sort the spans of a line in ascending horizontal direction. See comment
    in SortBlocks function.
    '''
    sspans = []
    for s in spans:
        x0 = str(int(s["bbox"][0] + 0.99999)).rjust(4,"0")
        sspans.append([x0, s])
    sspans.sort(key = takeFirst)
    return [s[1] for s in sspans]

def write_content(pdf_url, ofile):
    res = requests.get(pdf_url, proxies = proxy_dict)
    mem_area = res.content
    doc = fitz.open(stream=mem_area, filetype="pdf")

    # ifile = "sample_1.pdf"
    # ofile = ifile.replace(".pdf", ".txt")

    # doc = fitz.Document(ifile)
    # print(doc.metadata)

    toc = doc.getToC()

    # print(toc) #here, page nums are 1-indexed

    start_pages = []
    end_pages = []
    pages = doc.pageCount

    for idx, topic_list in enumerate(toc):
        if ('contents' in topic_list[1].lower()) or ('table of contents' in topic_list[1].lower()) or ('index' in topic_list[1].lower()):
            if idx == len(toc) - 1:
                start_page = toc[idx][2]
                end_page = pages
            else:
                start_page = toc[idx][2]
                end_page = toc[idx + 1][2] - 1 #1-indexed
            start_pages.append(start_page - 1) #making the content pages 0-indexed
            end_pages.append(end_page - 1) #making the content pages 0-indexed

    # print(start_pages, end_pages)


    fout = open(ofile,"wb")

    for i in range(pages):
        i_flag = 0
        for idx_ in range(len(start_pages)):
            if i>=start_pages[idx_] and i<=end_pages[idx_]:
                i_flag = 1
                break
        if i_flag == 1:
            continue
        pg_text = ""                                 # initialize page text buffer
        pg = doc.loadPage(i)                         # load page number i
        pgdict = pg.getText("dict")#, flags=fitz.TEXT_INHIBIT_SPACES)           # get dict out of it
        # print(text)
        # pgdict = json.loads(text)                    # create a dict out of it
        pg_dim = list(pg.MediaBox)
        width = pg_dim[2]
        blocks = SortBlocks(pgdict["blocks"], width)        # now re-arrange ... blocks
        for b in blocks:
            # print(b.keys())
            if "lines" not in b:
                continue
            lines = SortLines(b["lines"])            # ... lines
            for l in lines:
                spans = SortSpans(l["spans"])        # ... spans
                for s in spans:
                    # ensure that spans are separated by at least 1 blank
                    # (should make sense in most cases)
                    if pg_text.endswith(" ") or s["text"].startswith(" "):
                        pg_text += s["text"]
                    else:
                        pg_text += " " + s["text"]
                pg_text += " "                      # separate lines by newline
                pg_text = [item.replace("•", "").strip() for item in pg_text.split(" ") if (item!=" " and item!="")]
                pg_text = " ".join(pg_text)
            pg_text += "\n"

        pg_text = pg_text.encode(ENCODING, "ignore")

        # print(pg_text)
        fout.write(pg_text)

    fout.close()

#==============================================================================
# Main Program
#==============================================================================

with open("../data/new_pretrain_manuals/orig_to_new_fname.json", 'r') as f:
    fname_dict = json.load(f)

out_dir = "/mnt/e/manuals_dump"

if os.path.exists(out_dir) == False:
    os.mkdir(out_dir)

with open("../data/new_pretrain_manuals/all_download_links.txt", 'r') as f:
    for line in tqdm(f):
        link = line.strip()
        ofile = fname_dict[link.split("/")[-1]].replace(".pdf", ".txt")
        ofile = os.path.join(out_dir, ofile)
        write_content(link, ofile)

# pdf_url = "http://pdfstream.manualsonline.com/6/6e205ecf-8a6b-4e30-8ffb-08780ad42874.pdf"