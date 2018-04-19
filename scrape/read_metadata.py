import piexif, piexif.helper, json, pprint

filename = "photos/cbindonesia_1.jpg"
exif_dict = piexif.load(filename)
user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
data = json.loads(user_comment)

pprint.pprint(data)