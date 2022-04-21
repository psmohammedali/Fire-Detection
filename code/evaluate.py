import background_filter as bg_filter
import aimodel as aimodel
import and_frames as and_op
import fire_disorder_v2 as fire_disorder


if __name__ == "__main__":
    bg_filter.main()
    print("background filter done")
    aimodel.main()
    print("aimodel detection done")
    and_op.main()
    print("And Operation is done")
    fire_disorder.main()
    print("fire Disorder Done")
    print("Fire/Non-fire Detection is done successfully...")
    print("Detection are saved in output folder...")


