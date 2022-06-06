import pickle
import mysql.connector


class MySQLDBManager(object):
    """ MySQL database to organize experiment results """
    def __init__(self,
                 user='vasan',
                 host="localhost",     # Digital Storm Workstation
                 password='123',
                 database='experiments',
                 table='test_rl_suite',
                 readonly=False,
                 ssl_ca=None,
                 ):
        self.user = user
        self.host = host
        self.password = password
        self.database = database
        self.table = table
        self.readonly = readonly
        self.ssl_ca = ssl_ca
        self.run_id = None

        # Check if corresponding table exists, if not, create one
        if not self.check_table():
            self.create_table()

    def make_connection(self):
        try:
            # N.B: https://stackoverflow.com/questions/50557234/authentication-plugin-caching-sha2-password-is-not-supported
            config = {'user': self.user, 'host': self.host, 'password': self.password,
                      'database': self.database, 'auth_plugin': 'mysql_native_password'}
            if self.ssl_ca:
                config['ssl_ca'] = self.ssl_ca
                config['ssl_verify_cert'] = True
            cnx = mysql.connector.connect(**config)
            cursor = cnx.cursor()
        except Exception as e:
            print("{}: Could not connect to a database".format(e))
            raise e
        return cnx, cursor

    def check_table(self):
        cnx, cursor = self.make_connection()
        query = "SHOW TABLES LIKE '{}'".format(self.table)
        cursor.execute(query)
        result = cursor.fetchone()
        return result

    def create_table(self):
        if not self.readonly:
            cnx, cursor = self.make_connection()
            try:
                cursor.execute((
                    "CREATE TABLE `{}` ("
                    "  `id` int NOT NULL AUTO_INCREMENT," 
                    "  `description` TEXT,"
                    "  `cfg` LONGBLOB,"      
                    "  `episodic_returns` LONGBLOB,"
                    "  `episodic_lengths` LONGBLOB," 
                    "  `model` LONGBLOB,"
                    "  `metadata` LONGBLOB, "       
                    "  PRIMARY KEY (`id`)"
                    ") ENGINE=InnoDB".format(self.table)))
                cnx.commit()
                print("Created new table: {}".format(self.table))

            except Exception as e:
                print("{}: Could not create a new table".format(e))
            cnx.close()
        else:
            print('WARNING: Database readonly, cannot create a new table')

    def load_run(self, db_id):
        """ Load cfg column of table given db id

        Args:
            db_id (int): Unique database ID

        Returns:
            Tuple: (cfg, model, episodic_returns, episodic_lengths)

        """
        cnx, cursor = self.make_connection()
        cursor.execute(("SELECT cfg, model, episodic_returns, episodic_lengths, metadata FROM {} where id={}".format(
            self.table, db_id)))
        row = cursor.fetchall()[0]
        cnx.close()
        return (pickle.loads(r) if r is not None else None for r in row)

    def update(self, episodic_returns, episodic_lengths, model, metadata):
        if not self.readonly:
            assert self.run_id is not None, "Entry can't be updated before first save!"
            cnx, cursor = self.make_connection()
            query_string = "UPDATE {} SET model=%s," \
                           " episodic_returns=%s," \
                           " episodic_lengths=%s," \
                           " metadata=%s".format(self.table)
            values = [pickle.dumps(model), pickle.dumps(episodic_returns), pickle.dumps(episodic_lengths),
                      pickle.dumps(metadata)]

            query_string += " WHERE id=%s"
            values.append(self.run_id)
            cursor.execute(query_string, tuple(values))
            cnx.commit()
            cnx.close()
            print("Updated {} entry on row: {}".format(self.table, self.run_id))

    def save(self, cfg, episodic_returns, episodic_lengths, metadata):
        """

        Args:
            cfg:

        Returns:

        """
        if not self.readonly:
            cnx, cursor = self.make_connection()

            # Form a list of parameters that we want to save to database
            #   TODO: Poll this straight from db
            params_to_save = ['description', 'cfg', 'episodic_returns', 'episodic_lengths', 'metadata']
            values = [cfg['description'], pickle.dumps(cfg), pickle.dumps(episodic_returns),
                      pickle.dumps(episodic_lengths), pickle.dumps(metadata)]

            query = "INSERT INTO {} ({})" \
                    " VALUES ({})".format(self.table, ', '.join(params_to_save),
                                          ', '.join(["%s"] * len(params_to_save)))

            cursor.execute(query, values)
            cnx.commit()
            self.run_id = cursor.lastrowid
            cnx.close()
            print("Saved a new entry with ID {} into a database".format(self.run_id))
        else:
            print('Cannot save while in readonly mode')


class PickleExptSaver:
    """ Save pickle file with relevant experiment data """
    def __init__(self, filepath):
        self.filepath = filepath
        self.expt_data = {
            "cfg": None,
            "episodic_returns": None,
            "episodic_lengths": None,
            "metadata":None,
            "model": None,
        }

    def update(self, episodic_returns, episodic_lengths, model, metadata):
        self.expt_data["episodic_returns"] = episodic_returns
        self.expt_data["episodic_lengths"] = episodic_lengths
        self.expt_data["model"] = model
        self.expt_data["metadata"] = metadata
        self.pickle_dump()

    def save(self, cfg, episodic_returns, episodic_lengths, metadata):
        self.expt_data["cfg"] = cfg
        self.expt_data["episodic_returns"] = episodic_returns
        self.expt_data["episodic_lengths"] = episodic_lengths
        self.expt_data["metadata"] = metadata
        self.pickle_dump()

    def pickle_dump(self):
        with open(self.filepath, 'wb') as handle:
            pickle.dump(self.expt_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_run(self, filepath):
        with open(filepath, 'rb') as handle:
            expt_data = pickle.load(handle)

        return expt_data["cfg"], expt_data["model"], expt_data["episodic_returns"], \
               expt_data["episodic_lengths"], expt_data["metadata"]


if __name__ == '__main__':
    id = 56
    db = MySQLDBManager(host="localhost", user='vega', password='vector123')
    cfg, model, episodic_returns, episodic_lengths, metadata = db.load_run(db_id=id)

    # Create a duplicate record
    db.save(cfg, episodic_returns, episodic_lengths, metadata)
    db.update(episodic_returns, episodic_lengths, model, metadata)
    